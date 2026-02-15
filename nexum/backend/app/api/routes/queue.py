"""
Nexum API — Priority Queue Routes.

Manages the interruptible processing queue:
  - POST /queue/enqueue     — Add a URL to the queue
  - GET  /queue/status      — Full queue state
  - GET  /queue/job/{id}    — Single job status
  - POST /queue/cancel/{id} — Cancel a job
  - WS   /ws/queue          — Real-time queue event stream
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.services.queue.priority_queue_service import (
    JobPriority,
    priority_queue,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/queue", tags=["Queue"])


class EnqueueRequest(BaseModel):
    url: str = Field(..., description="Video, channel, or playlist URL")
    priority: str = Field("normal", description="'normal' or 'immediate'")
    job_type: Optional[str] = Field(None, description="Override auto-detection: 'video', 'channel', 'playlist'")


class EnqueueResponse(BaseModel):
    job_id: str
    url: str
    job_type: str
    priority: str
    status: str
    queue_position: Optional[int] = None


@router.post("/enqueue", response_model=EnqueueResponse)
async def enqueue_url(req: EnqueueRequest):
    """
    Add a URL to the processing queue.

    Set priority='immediate' to interrupt the current operation.
    Multiple immediate jobs stack — each interrupts the previous,
    and they drain in LIFO order before normal work resumes.
    """
    prio = JobPriority.IMMEDIATE if req.priority == "immediate" else JobPriority.NORMAL

    job = await priority_queue.enqueue(
        url=req.url,
        priority=prio,
        job_type=req.job_type,
    )

    # Compute queue position
    status = priority_queue.get_status()
    position = None
    if job.priority == JobPriority.NORMAL:
        for i, j in enumerate(status["normal_queue"]):
            if j["id"] == job.id:
                position = i + 1 + len(status["priority_stack"])
                break
    else:
        position = 0  # immediate = next up

    return EnqueueResponse(
        job_id=job.id,
        url=job.url,
        job_type=job.job_type,
        priority=job.priority.value,
        status=job.status.value,
        queue_position=position,
    )


@router.get("/status")
async def get_queue_status():
    """Get full queue state: current job, priority stack, normal queue, paused, completed."""
    return priority_queue.get_status()


@router.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    job = priority_queue.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a pending, paused, or running job."""
    job = await priority_queue.cancel(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job": job.to_dict()}


# ── WebSocket: Real-time queue events ────────────────────────────────

@router.websocket("/ws/queue")
async def queue_websocket(ws: WebSocket):
    """WebSocket endpoint for real-time queue event streaming."""
    await ws.accept()

    async def on_event(event):
        try:
            await ws.send_text(json.dumps(event, default=str))
        except Exception:
            pass

    priority_queue.add_listener(on_event)

    try:
        # Send initial state
        await ws.send_text(json.dumps({
            "type": "initial_state",
            "data": priority_queue.get_status(),
        }, default=str))

        while True:
            try:
                data = await ws.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug(f"Queue WS error: {e}")
    finally:
        priority_queue.remove_listener(on_event)
