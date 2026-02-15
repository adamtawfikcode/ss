"""
Nexum Priority Queue Service — Interruptible Task Queue.

Manages a two-tier queue (normal + priority) with pause/resume semantics:
  - Normal queue: Standard processing order
  - Priority queue: Interrupts current work, executes immediately, resumes prior work
  - Stackable: Multiple priority items form a LIFO stack that drains before normal work resumes

State machine per job:  PENDING → RUNNING → COMPLETED | FAILED
                                 → PAUSED (if interrupted by priority) → RUNNING
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobPriority(str, Enum):
    NORMAL = "normal"
    IMMEDIATE = "immediate"


@dataclass
class QueueJob:
    id: str
    url: str
    job_type: str  # "video" | "channel" | "playlist"
    priority: JobPriority
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    interrupted_by: Optional[str] = None  # job_id that interrupted this one
    resume_count: int = 0

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["priority"] = self.priority.value
        d["status"] = self.status.value
        return d


class PriorityQueueService:
    """
    Two-tier interruptible job queue.

    Architecture:
      ┌─────────────────────────────────────────┐
      │  Priority Stack (LIFO)                  │  ← drains first
      │  [job_C] [job_B]                        │
      ├─────────────────────────────────────────┤
      │  Normal Queue (FIFO)                    │  ← processes when stack empty
      │  [job_1] → [job_2] → [job_3]           │
      └─────────────────────────────────────────┘
      │  Currently Running: job_A (may be paused) │
      └───────────────────────────────────────────┘

    When a priority job arrives:
      1. Current running job gets status=PAUSED
      2. Priority job starts immediately
      3. On completion, check priority stack for more
      4. If stack empty, resume paused job from where it left off
    """

    def __init__(self):
        self._normal_queue: Deque[QueueJob] = deque()
        self._priority_stack: List[QueueJob] = []  # LIFO
        self._current_job: Optional[QueueJob] = None
        self._paused_jobs: List[QueueJob] = []  # stack of paused jobs
        self._completed: Deque[QueueJob] = deque(maxlen=200)
        self._all_jobs: Dict[str, QueueJob] = {}
        self._lock = asyncio.Lock()
        self._processing = False
        self._process_task: Optional[asyncio.Task] = None
        self._interrupt_event = asyncio.Event()
        self._listeners: List[Callable] = []

    # ── Public API ───────────────────────────────────────────────────

    async def enqueue(
        self,
        url: str,
        priority: JobPriority = JobPriority.NORMAL,
        job_type: Optional[str] = None,
    ) -> QueueJob:
        """Add a URL to the processing queue."""
        if not job_type:
            job_type = self._detect_type(url)

        job = QueueJob(
            id=str(uuid.uuid4()),
            url=url,
            job_type=job_type,
            priority=priority,
        )

        async with self._lock:
            self._all_jobs[job.id] = job

            if priority == JobPriority.IMMEDIATE:
                self._priority_stack.append(job)
                logger.info(f"Priority job enqueued: {job.id} ({url})")
                # Signal interrupt if something is running
                if self._current_job and self._current_job.status == JobStatus.RUNNING:
                    self._interrupt_event.set()
            else:
                self._normal_queue.append(job)
                logger.info(f"Normal job enqueued: {job.id} ({url})")

        await self._notify({"type": "job_enqueued", "job": job.to_dict()})

        # Auto-start processing if not already running
        if not self._processing:
            self._start_processing()

        return job

    async def cancel(self, job_id: str) -> Optional[QueueJob]:
        """Cancel a pending or running job."""
        async with self._lock:
            job = self._all_jobs.get(job_id)
            if not job:
                return None

            if job.status in (JobStatus.PENDING, JobStatus.PAUSED):
                job.status = JobStatus.CANCELLED
                # Remove from queues
                if job in self._normal_queue:
                    self._normal_queue.remove(job)
                if job in self._priority_stack:
                    self._priority_stack.remove(job)
                if job in self._paused_jobs:
                    self._paused_jobs.remove(job)
            elif job.status == JobStatus.RUNNING:
                job.status = JobStatus.CANCELLED
                self._interrupt_event.set()

        await self._notify({"type": "job_cancelled", "job": job.to_dict()})
        return job

    def get_status(self) -> Dict[str, Any]:
        """Full queue state snapshot."""
        return {
            "is_processing": self._processing,
            "current_job": self._current_job.to_dict() if self._current_job else None,
            "priority_stack": [j.to_dict() for j in reversed(self._priority_stack)],
            "normal_queue": [j.to_dict() for j in self._normal_queue],
            "paused_jobs": [j.to_dict() for j in self._paused_jobs],
            "completed_recent": [j.to_dict() for j in list(self._completed)[-20:]],
            "stats": {
                "total_enqueued": len(self._all_jobs),
                "pending_normal": len(self._normal_queue),
                "pending_priority": len(self._priority_stack),
                "paused": len(self._paused_jobs),
                "completed": sum(1 for j in self._all_jobs.values() if j.status == JobStatus.COMPLETED),
                "failed": sum(1 for j in self._all_jobs.values() if j.status == JobStatus.FAILED),
            },
        }

    def get_job(self, job_id: str) -> Optional[Dict]:
        job = self._all_jobs.get(job_id)
        return job.to_dict() if job else None

    def add_listener(self, callback: Callable):
        """Register a listener for queue events (for WebSocket broadcast)."""
        self._listeners.append(callback)

    def remove_listener(self, callback: Callable):
        self._listeners = [l for l in self._listeners if l is not callback]

    # ── Processing Loop ──────────────────────────────────────────────

    def _start_processing(self):
        if self._process_task and not self._process_task.done():
            return
        self._process_task = asyncio.create_task(self._process_loop())

    async def _process_loop(self):
        """Main processing loop — handles priority interrupts."""
        self._processing = True
        try:
            while True:
                job = await self._next_job()
                if not job:
                    break

                await self._execute_job(job)

        except Exception as e:
            logger.error(f"Queue processing error: {e}")
        finally:
            self._processing = False

    async def _next_job(self) -> Optional[QueueJob]:
        """Get next job: priority stack > paused jobs > normal queue."""
        async with self._lock:
            # 1. Priority stack first (LIFO)
            if self._priority_stack:
                return self._priority_stack.pop()

            # 2. Resume paused jobs
            if self._paused_jobs:
                job = self._paused_jobs.pop()
                job.resume_count += 1
                return job

            # 3. Normal queue (FIFO)
            if self._normal_queue:
                return self._normal_queue.popleft()

        return None

    async def _execute_job(self, job: QueueJob):
        """Execute a single job with interrupt support."""
        self._current_job = job
        job.status = JobStatus.RUNNING
        if not job.started_at:
            job.started_at = time.time()
        self._interrupt_event.clear()

        await self._notify({"type": "job_started", "job": job.to_dict()})

        try:
            result = await self._run_with_interrupt(job)

            if job.status == JobStatus.CANCELLED:
                return

            if job.status == JobStatus.PAUSED:
                # Was interrupted — it's already in paused_jobs
                return

            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.completed_at = time.time()
            job.result = result
            self._completed.append(job)

            await self._notify({"type": "job_completed", "job": job.to_dict()})

        except Exception as e:
            if job.status != JobStatus.CANCELLED:
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = time.time()
                self._completed.append(job)
                await self._notify({"type": "job_failed", "job": job.to_dict()})
        finally:
            if self._current_job == job:
                self._current_job = None

    async def _run_with_interrupt(self, job: QueueJob) -> Optional[Dict]:
        """
        Execute the actual processing, checking for priority interrupts.

        In production this calls crawler_service + media_service.
        The interrupt check happens between pipeline stages.
        """
        stages = self._get_stages(job)
        start_stage = 0

        # If resuming, skip already-completed stages
        if job.resume_count > 0:
            start_stage = int(job.progress * len(stages))
            job.message = f"Resuming from stage {start_stage + 1}/{len(stages)}"
            await self._notify({"type": "job_resumed", "job": job.to_dict()})

        for i in range(start_stage, len(stages)):
            # Check for interrupt before each stage
            if self._interrupt_event.is_set():
                self._interrupt_event.clear()

                if job.status == JobStatus.CANCELLED:
                    return None

                # Pause this job and let priority job run
                async with self._lock:
                    if self._priority_stack:
                        job.status = JobStatus.PAUSED
                        job.progress = i / max(len(stages), 1)
                        priority_job = self._priority_stack[-1]
                        job.interrupted_by = priority_job.id
                        job.message = f"Paused at stage {i+1}/{len(stages)} — priority job {priority_job.id[:8]}"
                        self._paused_jobs.append(job)
                        self._current_job = None
                        await self._notify({"type": "job_paused", "job": job.to_dict()})
                        return None

            # Execute stage
            stage_name, stage_fn = stages[i]
            job.message = f"Stage {i+1}/{len(stages)}: {stage_name}"
            job.progress = i / max(len(stages), 1)
            await self._notify({"type": "job_progress", "job": job.to_dict()})

            await stage_fn(job)

        return {"stages_completed": len(stages), "url": job.url, "type": job.job_type}

    def _get_stages(self, job: QueueJob) -> List[tuple]:
        """Define processing stages per job type."""
        if job.job_type == "video":
            return [
                ("Fetching metadata", self._stage_fetch_metadata),
                ("Downloading media", self._stage_download),
                ("Transcribing audio", self._stage_transcribe),
                ("Extracting frames", self._stage_extract_frames),
                ("Running OCR", self._stage_ocr),
                ("Computing embeddings", self._stage_embeddings),
                ("Audio analysis", self._stage_audio_analysis),
                ("Graph sync", self._stage_graph_sync),
                ("Indexing", self._stage_index),
            ]
        elif job.job_type == "channel":
            return [
                ("Fetching channel info", self._stage_fetch_metadata),
                ("Discovering videos", self._stage_discover_videos),
                ("Queueing videos", self._stage_queue_discovered),
            ]
        elif job.job_type == "playlist":
            return [
                ("Fetching playlist info", self._stage_fetch_metadata),
                ("Listing playlist items", self._stage_discover_videos),
                ("Queueing videos", self._stage_queue_discovered),
            ]
        return [("Processing", self._stage_fetch_metadata)]

    # ── Pipeline Stages (real + demo stubs) ──────────────────────────

    async def _stage_fetch_metadata(self, job: QueueJob):
        """Fetch metadata via yt-dlp (or simulate in demo mode)."""
        await asyncio.sleep(0.8)  # simulate

    async def _stage_download(self, job: QueueJob):
        await asyncio.sleep(1.5)

    async def _stage_transcribe(self, job: QueueJob):
        await asyncio.sleep(2.0)

    async def _stage_extract_frames(self, job: QueueJob):
        await asyncio.sleep(1.0)

    async def _stage_ocr(self, job: QueueJob):
        await asyncio.sleep(0.8)

    async def _stage_embeddings(self, job: QueueJob):
        await asyncio.sleep(1.2)

    async def _stage_audio_analysis(self, job: QueueJob):
        await asyncio.sleep(1.5)

    async def _stage_graph_sync(self, job: QueueJob):
        await asyncio.sleep(0.5)

    async def _stage_index(self, job: QueueJob):
        await asyncio.sleep(0.3)

    async def _stage_discover_videos(self, job: QueueJob):
        await asyncio.sleep(2.0)

    async def _stage_queue_discovered(self, job: QueueJob):
        await asyncio.sleep(0.5)

    # ── Utilities ────────────────────────────────────────────────────

    @staticmethod
    def _detect_type(url: str) -> str:
        url_lower = url.lower()
        if "playlist" in url_lower or "list=" in url_lower:
            return "playlist"
        if any(p in url_lower for p in ["/@", "/c/", "/channel/", "/user/"]):
            return "channel"
        return "video"

    async def _notify(self, event: Dict):
        """Notify all listeners of a queue event."""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event)
                else:
                    listener(event)
            except Exception as e:
                logger.debug(f"Listener error: {e}")


# Module singleton
priority_queue = PriorityQueueService()
