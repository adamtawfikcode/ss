"""
Nexum API — Comment Routes

Endpoints for comment browsing, thread retrieval, and ingestion triggers.
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import Comment, CommentAuthor, CommentStatus, Entity
from app.schemas.schemas import (
    CommentAuthorSchema,
    CommentIngestRequest,
    CommentIngestResponse,
    CommentSchema,
    EntitySchema,
)

router = APIRouter(prefix="/comments", tags=["Comments"])


@router.get("/{video_id}", response_model=List[CommentSchema])
async def list_video_comments(
    video_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: Optional[str] = None,
    sort: str = Query("likes", pattern="^(likes|newest|oldest|sentiment)$"),
    db: AsyncSession = Depends(get_db),
):
    """List comments for a video with sorting and pagination."""
    try:
        vid_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID")

    query = select(Comment).where(Comment.video_id == vid_uuid)

    if status:
        query = query.where(Comment.status == status)
    else:
        query = query.where(Comment.status.in_([CommentStatus.PROCESSED, CommentStatus.PENDING]))

    if sort == "likes":
        query = query.order_by(Comment.like_count.desc())
    elif sort == "newest":
        query = query.order_by(Comment.timestamp_posted.desc())
    elif sort == "oldest":
        query = query.order_by(Comment.timestamp_posted.asc())
    elif sort == "sentiment":
        query = query.order_by(Comment.sentiment_score.desc())

    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    comments = result.scalars().all()

    schemas = []
    for c in comments:
        author_name = None
        if c.author_id:
            author = await db.get(CommentAuthor, c.author_id)
            author_name = author.display_name if author else None

        schemas.append(CommentSchema(
            id=str(c.id),
            platform_comment_id=c.platform_comment_id,
            video_id=str(c.video_id),
            author_display_name=author_name,
            parent_comment_id=str(c.parent_comment_id) if c.parent_comment_id else None,
            root_thread_id=str(c.root_thread_id) if c.root_thread_id else None,
            depth_level=c.depth_level,
            child_count=c.child_count,
            text=c.text,
            like_count=c.like_count,
            reply_count=c.reply_count,
            language=c.language,
            sentiment_score=c.sentiment_score,
            sentiment_label=c.sentiment_label,
            toxicity_score=c.toxicity_score,
            topic_labels=c.topic_labels,
            entities=c.entities_json,
            extracted_timestamps=c.extracted_timestamps,
            status=c.status.value,
            timestamp_posted=c.timestamp_posted,
        ))

    return schemas


@router.post("/ingest", response_model=CommentIngestResponse)
async def trigger_comment_ingestion(
    request: CommentIngestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Trigger comment ingestion for a specific video."""
    from app.workers.tasks import ingest_comments_task
    ingest_comments_task.delay(request.video_id, request.max_comments, request.max_reply_depth)
    return CommentIngestResponse(
        video_id=request.video_id,
        comments_fetched=0,
        comments_new=0,
        comments_spam_filtered=0,
        authors_discovered=0,
        status="queued",
    )


@router.get("/stats/summary")
async def comment_stats(db: AsyncSession = Depends(get_db)):
    """Get comment system statistics."""
    total = await db.scalar(select(func.count(Comment.id))) or 0
    processed = await db.scalar(select(func.count(Comment.id)).where(Comment.status == CommentStatus.PROCESSED)) or 0
    pending = await db.scalar(select(func.count(Comment.id)).where(Comment.status == CommentStatus.PENDING)) or 0
    spam = await db.scalar(select(func.count(Comment.id)).where(Comment.status == CommentStatus.SPAM)) or 0
    toxic = await db.scalar(select(func.count(Comment.id)).where(Comment.status == CommentStatus.TOXIC)) or 0
    total_authors = await db.scalar(select(func.count(CommentAuthor.id))) or 0
    total_entities = await db.scalar(select(func.count(Entity.id))) or 0

    avg_depth = await db.scalar(select(func.avg(Comment.depth_level))) or 0.0
    max_depth = await db.scalar(select(func.max(Comment.depth_level))) or 0

    return {
        "total_comments": total,
        "processed": processed,
        "pending": pending,
        "spam_filtered": spam,
        "toxic_filtered": toxic,
        "total_authors": total_authors,
        "total_entities": total_entities,
        "avg_thread_depth": round(float(avg_depth), 2),
        "max_thread_depth": max_depth,
        "spam_rate": round(spam / max(total, 1), 4),
    }


# ── Authors ──────────────────────────────────────────────────────────────

@router.get("/authors/top", response_model=List[CommentAuthorSchema])
async def top_authors(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """Get most active comment authors."""
    result = await db.execute(
        select(CommentAuthor)
        .where(CommentAuthor.is_pruned == False)
        .order_by(CommentAuthor.comment_count.desc())
        .limit(limit)
    )
    authors = result.scalars().all()
    return [
        CommentAuthorSchema(
            id=str(a.id),
            display_name=a.display_name,
            platform=a.platform.value,
            comment_count=a.comment_count,
            first_seen_at=a.first_seen_at,
            last_seen_at=a.last_seen_at,
            sentiment_distribution=a.sentiment_distribution,
        )
        for a in authors
    ]


# ── Entities ─────────────────────────────────────────────────────────────

@router.get("/entities/top", response_model=List[EntitySchema])
async def top_entities(
    limit: int = Query(30, ge=1, le=200),
    entity_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get most-mentioned entities."""
    query = select(Entity).order_by(Entity.mention_count.desc()).limit(limit)
    if entity_type:
        query = query.where(Entity.entity_type == entity_type)
    result = await db.execute(query)
    entities = result.scalars().all()
    return [
        EntitySchema(
            id=str(e.id),
            canonical_name=e.canonical_name,
            entity_type=e.entity_type.value,
            mention_count=e.mention_count,
            aliases=e.aliases,
            first_seen_at=e.first_seen_at,
        )
        for e in entities
    ]
