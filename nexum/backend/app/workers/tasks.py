"""
Nexum Celery Worker Tasks

Asynchronous task definitions for:
- Video processing pipeline
- Crawl cycles
- Re-indexing
- Feedback recalibration
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

# ── Celery App ───────────────────────────────────────────────────────────

celery_app = Celery(
    "nexum",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=3600,   # 1 hour soft limit
    task_time_limit=7200,        # 2 hour hard limit
    task_default_queue="default",
    task_routes={
        "app.workers.tasks.process_video_task": {"queue": "processing"},
        "app.workers.tasks.crawl_cycle_task": {"queue": "crawler"},
        "app.workers.tasks.reindex_video_task": {"queue": "processing"},
        "app.workers.tasks.ingest_comments_task": {"queue": "comments"},
        "app.workers.tasks.process_comments_task": {"queue": "comments"},
        "app.workers.tasks.sync_graph_task": {"queue": "graph"},
        "app.workers.tasks.update_author_profiles_task": {"queue": "graph"},
        "app.workers.tasks.detect_audience_overlap_task": {"queue": "graph"},
        # v4: calibration
        "app.workers.tasks.fit_calibration_task": {"queue": "default"},
    },
    # For sequential GPU: only 1 processing task at a time
    worker_concurrency=1,
)

# ── Periodic Tasks ───────────────────────────────────────────────────────

celery_app.conf.beat_schedule = {
    "crawl-every-5-minutes": {
        "task": "app.workers.tasks.crawl_cycle_task",
        "schedule": settings.crawler_interval_seconds,
    },
    "recalibrate-weights-daily": {
        "task": "app.workers.tasks.recalibrate_weights_task",
        "schedule": crontab(hour=3, minute=0),  # 3 AM daily
    },
    "health-check-every-minute": {
        "task": "app.workers.tasks.health_check_task",
        "schedule": 60.0,
    },
    # ── Social Knowledge Graph Tasks (v3) ────────────────────────────
    "ingest-comments-every-10-minutes": {
        "task": "app.workers.tasks.ingest_comments_cycle_task",
        "schedule": settings.comment_ingest_interval_seconds,
    },
    "process-pending-comments-every-2-minutes": {
        "task": "app.workers.tasks.process_comments_task",
        "schedule": 120.0,
    },
    "update-author-profiles-hourly": {
        "task": "app.workers.tasks.update_author_profiles_task",
        "schedule": crontab(minute=30),  # every hour at :30
    },
    "detect-audience-overlap-daily": {
        "task": "app.workers.tasks.detect_audience_overlap_task",
        "schedule": crontab(hour=4, minute=0),  # 4 AM daily
    },
    # v4: confidence calibration
    "fit-calibration-daily": {
        "task": "app.workers.tasks.fit_calibration_task",
        "schedule": crontab(hour=2, minute=30),  # 2:30 AM daily
    },
}


# ── Helpers ──────────────────────────────────────────────────────────────

def run_async(coro):
    """Run an async coroutine from sync Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Tasks ────────────────────────────────────────────────────────────────

@celery_app.task(
    name="app.workers.tasks.process_video_task",
    bind=True,
    max_retries=3,
    default_retry_delay=60,
    acks_late=True,
)
def process_video_task(self, video_id: str):
    """Process a single video through the full ML pipeline."""
    try:
        logger.info(f"Processing video: {video_id}")
        from app.services.media.media_service import media_service
        run_async(media_service.process_video(video_id))
        logger.info(f"Completed processing: {video_id}")
    except Exception as exc:
        logger.error(f"Task failed for {video_id}: {exc}")
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@celery_app.task(name="app.workers.tasks.crawl_cycle_task")
def crawl_cycle_task():
    """Run one crawl cycle across all channels/accounts."""
    try:
        logger.info("Starting crawl cycle")
        from app.services.crawler.crawler_service import crawler_service
        run_async(crawler_service.run_crawl_cycle())
        logger.info("Crawl cycle complete")
    except Exception as e:
        logger.error(f"Crawl cycle failed: {e}")


@celery_app.task(
    name="app.workers.tasks.reindex_video_task",
    bind=True,
    max_retries=2,
)
def reindex_video_task(self, video_id: str, model_version: Optional[str] = None):
    """Re-process and re-index a video with updated models."""
    try:
        logger.info(f"Re-indexing video: {video_id}")
        from app.services.media.media_service import media_service
        run_async(media_service.process_video(video_id))
    except Exception as exc:
        logger.error(f"Re-index failed for {video_id}: {exc}")
        raise self.retry(exc=exc, countdown=120)


@celery_app.task(name="app.workers.tasks.recalibrate_weights_task")
def recalibrate_weights_task():
    """Analyze user feedback and adjust fusion weights."""
    try:
        logger.info("Recalibrating fusion weights from feedback")
        from app.services.feedback.feedback_service import feedback_service
        run_async(feedback_service.recalibrate_weights())
    except Exception as e:
        logger.error(f"Weight recalibration failed: {e}")


@celery_app.task(name="app.workers.tasks.health_check_task")
def health_check_task():
    """Periodic health check — ensures workers are alive."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


# ═══════════════════════════════════════════════════════════════════════════
# Social Knowledge Graph Tasks (v3)
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(
    name="app.workers.tasks.ingest_comments_task",
    bind=True,
    max_retries=3,
    default_retry_delay=120,
    acks_late=True,
)
def ingest_comments_task(self, video_id: str, max_comments: int = 200, max_reply_depth: int = 20):
    """Ingest comments for a single video."""
    try:
        logger.info(f"Ingesting comments for video: {video_id}")
        from app.services.comments.comment_ingestion_service import comment_ingestion_service
        from app.core.database import async_session_factory

        async def _run():
            async with async_session_factory() as db:
                return await comment_ingestion_service.ingest_comments_for_video(
                    video_id, db, max_comments, max_reply_depth,
                )

        stats = run_async(_run())
        logger.info(f"Comment ingestion done for {video_id}: {stats}")
        return stats
    except Exception as exc:
        logger.error(f"Comment ingestion failed for {video_id}: {exc}")
        raise self.retry(exc=exc, countdown=120 * (2 ** self.request.retries))


@celery_app.task(name="app.workers.tasks.ingest_comments_cycle_task")
def ingest_comments_cycle_task():
    """Periodic task: ingest comments for all recently indexed videos."""
    try:
        logger.info("Starting comment ingestion cycle")
        from app.models.models import Video, VideoStatus
        from app.core.database import async_session_factory
        from sqlalchemy import select

        async def _get_video_ids():
            async with async_session_factory() as db:
                result = await db.execute(
                    select(Video.id)
                    .where(Video.status == VideoStatus.INDEXED)
                    .order_by(Video.updated_at.desc())
                    .limit(20)
                )
                return [str(r[0]) for r in result.all()]

        video_ids = run_async(_get_video_ids())
        for vid_id in video_ids:
            ingest_comments_task.delay(vid_id)

        logger.info(f"Queued comment ingestion for {len(video_ids)} videos")
    except Exception as e:
        logger.error(f"Comment ingestion cycle failed: {e}")


@celery_app.task(name="app.workers.tasks.process_comments_task")
def process_comments_task():
    """Process pending comments through NLP pipeline."""
    try:
        logger.info("Processing pending comments")
        from app.services.comments.comment_processing_service import comment_processing_service
        from app.core.database import async_session_factory

        async def _run():
            async with async_session_factory() as db:
                stats = await comment_processing_service.process_pending_comments(db)
                await db.commit()
                return stats

        stats = run_async(_run())
        logger.info(f"Comment processing complete: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Comment processing failed: {e}")


@celery_app.task(name="app.workers.tasks.update_author_profiles_task")
def update_author_profiles_task():
    """Aggregate comment data into author profiles."""
    try:
        logger.info("Updating author profiles")
        from app.services.comments.comment_processing_service import comment_processing_service
        from app.core.database import async_session_factory

        async def _run():
            async with async_session_factory() as db:
                await comment_processing_service.update_author_profiles(db)
                await db.commit()

        run_async(_run())
        logger.info("Author profiles updated")
    except Exception as e:
        logger.error(f"Author profile update failed: {e}")


@celery_app.task(name="app.workers.tasks.detect_audience_overlap_task")
def detect_audience_overlap_task():
    """Detect shared audiences between channels."""
    try:
        logger.info("Detecting audience overlap")
        from app.services.graph.graph_service import graph_service
        run_async(graph_service.detect_audience_overlap())
        logger.info("Audience overlap detection complete")
    except Exception as e:
        logger.error(f"Audience overlap detection failed: {e}")


@celery_app.task(name="app.workers.tasks.sync_graph_task")
def sync_graph_task(video_id: str):
    """Sync a video's data (comments, entities, authors) to Neo4j."""
    try:
        logger.info(f"Syncing graph for video: {video_id}")
        from app.services.graph.graph_service import graph_service
        from app.models.models import Video, Comment, CommentAuthor, CommentStatus
        from app.core.database import async_session_factory
        from sqlalchemy import select

        async def _run():
            async with async_session_factory() as db:
                video = await db.get(Video, video_id)
                if not video:
                    return

                # Sync video node
                await graph_service.upsert_video_node(str(video.id), {
                    "title": video.title,
                    "platform": video.platform.value if video.platform else "other",
                    "url": video.url,
                    "view_count": video.view_count or 0,
                })

                # Sync channel → video edge
                if video.channel_id:
                    await graph_service.link_channel_uploaded(str(video.channel_id), str(video.id))

                # Sync processed comments
                result = await db.execute(
                    select(Comment).where(
                        Comment.video_id == video.id,
                        Comment.status == CommentStatus.PROCESSED,
                    )
                )
                for comment in result.scalars().all():
                    await graph_service.upsert_comment_node(str(comment.id), {
                        "text": comment.text[:500],
                        "video_id": str(comment.video_id),
                        "depth_level": comment.depth_level,
                        "sentiment_score": comment.sentiment_score,
                        "like_count": comment.like_count,
                    })
                    await graph_service.link_comment_to_video(str(comment.id), str(video.id))

                    if comment.author_id:
                        author = await db.get(CommentAuthor, comment.author_id)
                        if author:
                            await graph_service.upsert_author_node(str(author.id), {
                                "display_name": author.display_name,
                                "comment_count": author.comment_count,
                            })
                            await graph_service.link_author_wrote(str(author.id), str(comment.id))
                            await graph_service.link_author_commented_on(str(author.id), str(video.id))

                    if comment.parent_comment_id:
                        await graph_service.link_comment_replies_to(
                            str(comment.id), str(comment.parent_comment_id),
                        )

                    # Entity mentions
                    for entity_name in (comment.entities_json or []):
                        await graph_service.upsert_entity_node(entity_name.lower(), {
                            "entity_type": "other",
                        })
                        await graph_service.link_comment_mentions_entity(
                            str(comment.id), entity_name.lower(),
                        )

        run_async(_run())
        logger.info(f"Graph sync complete for video: {video_id}")
    except Exception as e:
        logger.error(f"Graph sync failed for {video_id}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Calibrated Intelligence Tasks (v4)
# ═══════════════════════════════════════════════════════════════════════════

@celery_app.task(name="app.workers.tasks.fit_calibration_task")
def fit_calibration_task():
    """
    Daily calibration fitting.

    Collects user feedback + manual labels accumulated since last fit,
    trains Platt/Isotonic calibrators for each model target, persists to disk.
    """
    try:
        logger.info("Starting calibration fitting")
        from app.ml.calibration import calibration_service

        results = calibration_service.fit_all()
        calibration_service.save()

        fitted = sum(1 for v in results.values() if v)
        logger.info(f"Calibration fit complete: {fitted}/{len(results)} targets fitted")
        return {"fitted": fitted, "total": len(results), "details": results}
    except Exception as e:
        logger.error(f"Calibration fitting failed: {e}")

