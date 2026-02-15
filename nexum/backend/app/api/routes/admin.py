"""
Nexum API — Admin routes.
"""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.models import (
    AcousticChangePoint,
    AudioSegment,
    Comment,
    CommentAuthor,
    CommunityPost,
    Entity,
    EvaluationMetric,
    Feedback,
    FeedbackType,
    Frame,
    ModelVersion,
    Playlist,
    Segment,
    Topic,
    TranscriptAlignment,
    Video,
    VideoStatus,
)
from app.schemas.schemas import (
    EvaluationMetricSchema,
    FusionWeightsUpdate,
    ModelVersionSchema,
    ReindexRequest,
    SystemMetrics,
)
from app.services.search.vector_store import vector_store

settings = get_settings()
router = APIRouter(prefix="/admin", tags=["Admin"])


@router.get("/metrics", response_model=SystemMetrics)
async def get_system_metrics(db: AsyncSession = Depends(get_db)):
    """System-wide metrics for admin dashboard."""
    total_videos = await db.scalar(select(func.count(Video.id))) or 0
    indexed = await db.scalar(
        select(func.count(Video.id)).where(Video.status == VideoStatus.INDEXED)
    ) or 0
    queued = await db.scalar(
        select(func.count(Video.id)).where(Video.status == VideoStatus.QUEUED)
    ) or 0
    failed = await db.scalar(
        select(func.count(Video.id)).where(Video.status == VideoStatus.FAILED)
    ) or 0
    total_segments = await db.scalar(select(func.count(Segment.id))) or 0
    total_frames = await db.scalar(select(func.count(Frame.id))) or 0
    total_feedback = await db.scalar(select(func.count(Feedback.id))) or 0
    upvotes = await db.scalar(
        select(func.count(Feedback.id)).where(Feedback.feedback_type == FeedbackType.UPVOTE)
    ) or 0
    downvotes = await db.scalar(
        select(func.count(Feedback.id)).where(Feedback.feedback_type == FeedbackType.DOWNVOTE)
    ) or 0

    upvote_ratio = upvotes / max(upvotes + downvotes, 1)

    # Extended stats
    total_comments = await db.scalar(select(func.count(Comment.id))) or 0
    total_entities = await db.scalar(select(func.count(Entity.id))) or 0
    total_authors = await db.scalar(select(func.count(CommentAuthor.id))) or 0
    total_topics = await db.scalar(select(func.count(Topic.id))) or 0
    total_playlists = await db.scalar(select(func.count(Playlist.id))) or 0
    total_community_posts = await db.scalar(select(func.count(CommunityPost.id))) or 0
    total_audio_segments = await db.scalar(select(func.count(AudioSegment.id))) or 0
    total_change_points = await db.scalar(select(func.count(AcousticChangePoint.id))) or 0
    total_alignment_warnings = await db.scalar(
        select(func.count(TranscriptAlignment.id)).where(
            TranscriptAlignment.quality_level.in_(["poor", "mismatch"])
        )
    ) or 0
    avg_alignment = await db.scalar(select(func.avg(TranscriptAlignment.alignment_score))) or 0.0

    # Calibration version
    from app.ml.calibration import calibration_service
    cal_version = calibration_service.version

    # Vector store stats
    vs_stats = vector_store.get_collection_stats()

    return SystemMetrics(
        total_videos=total_videos,
        indexed_videos=indexed,
        queued_videos=queued,
        failed_videos=failed,
        total_segments=total_segments,
        total_frames=total_frames,
        total_feedback=total_feedback,
        upvote_ratio=round(upvote_ratio, 4),
        avg_search_latency_ms=0.0,
        worker_count=0,
        queue_depth=queued,
        storage_used_gb=0.0,
        total_comments=total_comments,
        total_entities=total_entities,
        total_comment_authors=total_authors,
        total_topics=total_topics,
        total_playlists=total_playlists,
        total_community_posts=total_community_posts,
        graph_edges=0,
        total_audio_segments=total_audio_segments,
        total_change_points=total_change_points,
        total_alignment_warnings=total_alignment_warnings,
        avg_alignment_score=round(float(avg_alignment), 3),
        calibration_version=cal_version,
    )


@router.post("/reindex")
async def trigger_reindex(
    request: ReindexRequest, db: AsyncSession = Depends(get_db)
):
    """Trigger re-indexing of videos."""
    from app.workers.tasks import reindex_video_task

    if request.video_ids:
        for vid_id in request.video_ids:
            reindex_video_task.delay(vid_id, request.model_version)
        return {"status": "queued", "count": len(request.video_ids)}

    # Re-index all indexed videos
    result = await db.execute(
        select(Video.id).where(Video.status == VideoStatus.INDEXED)
    )
    video_ids = [str(r[0]) for r in result.all()]
    for vid_id in video_ids:
        reindex_video_task.delay(vid_id, request.model_version)

    return {"status": "queued", "count": len(video_ids)}


@router.get("/models", response_model=List[ModelVersionSchema])
async def list_models(db: AsyncSession = Depends(get_db)):
    """List all registered model versions."""
    result = await db.execute(
        select(ModelVersion).order_by(ModelVersion.created_at.desc())
    )
    models = result.scalars().all()
    return [
        ModelVersionSchema(
            id=str(m.id),
            name=m.name,
            version=m.version,
            model_type=m.model_type,
            is_active=m.is_active,
            accuracy_metrics=m.accuracy_metrics,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.get("/weights")
async def get_fusion_weights():
    """Get current multimodal fusion weights."""
    weight_fields = [f for f in dir(settings) if f.startswith("weight_")]
    return {f.replace("weight_", ""): getattr(settings, f) for f in weight_fields}


@router.put("/weights")
async def update_fusion_weights(update: FusionWeightsUpdate):
    """Update any of the 14 multimodal fusion weights at runtime."""
    updated = {}
    for field_name, value in update.model_dump(exclude_none=True).items():
        if hasattr(settings, field_name):
            setattr(settings, field_name, value)
            updated[field_name.replace("weight_", "")] = value
    return {"status": "updated", "weights": updated}


@router.get("/evaluation", response_model=List[EvaluationMetricSchema])
async def get_evaluation_metrics(
    limit: int = 100, db: AsyncSession = Depends(get_db)
):
    """Get evaluation metrics history."""
    result = await db.execute(
        select(EvaluationMetric)
        .order_by(EvaluationMetric.measured_at.desc())
        .limit(limit)
    )
    metrics = result.scalars().all()
    return [
        EvaluationMetricSchema(
            metric_name=m.metric_name,
            metric_value=m.metric_value,
            model_version=None,
            measured_at=m.measured_at,
        )
        for m in metrics
    ]


@router.get("/vector-stats")
async def get_vector_stats():
    """Get Qdrant collection statistics."""
    return vector_store.get_collection_stats()
