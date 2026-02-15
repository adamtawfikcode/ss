"""
Nexum Recrawl Service — Intelligent re-crawling with change detection.

When a URL is submitted that already exists in the database:
1. Detect it's a duplicate via platform_id
2. Fetch fresh metadata from YouTube
3. Diff every field against stored values
4. Update changed fields, create RecrawlEvent
5. Re-ingest new comments (only those posted after last_crawled_at)
6. Recompute signals that depend on changed data
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import (
    Channel, Comment, RecrawlEvent, Video, VideoStatus,
)

logger = logging.getLogger(__name__)


# Fields we track for change detection on Video
VIDEO_DIFF_FIELDS = [
    "title", "description", "view_count", "like_count", "dislike_count",
    "comment_count", "tags", "categories", "thumbnail_url", "duration_seconds",
    "live_status", "chapters_json", "captions_info", "age_limit", "is_short",
    "has_captions", "caption_languages", "has_auto_captions", "auto_caption_languages",
]

CHANNEL_DIFF_FIELDS = [
    "name", "description", "subscriber_count", "total_videos",
    "banner_url", "country", "custom_url",
]


class RecrawlService:
    """Handles re-crawling of existing videos/channels with change tracking."""

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Public API ───────────────────────────────────────────────────

    async def check_and_recrawl_video(
        self,
        platform_id: str,
        fresh_data: Dict[str, Any],
        trigger: str = "input_duplicate",
    ) -> Optional[Dict[str, Any]]:
        """
        Check if video exists. If yes, diff and update.
        Returns recrawl summary or None if video is new.
        """
        result = await self.db.execute(
            select(Video).where(Video.platform_id == platform_id)
        )
        existing = result.scalar_one_or_none()

        if existing is None:
            return None  # New video, proceed with normal crawl

        logger.info(f"Recrawl detected for video {platform_id} (trigger={trigger})")

        # Compute diff
        changes = self._diff_video(existing, fresh_data)
        if not changes["fields_changed"]:
            # Nothing changed, just bump last_crawled_at
            existing.last_crawled_at = datetime.now(timezone.utc)
            existing.updated_at = datetime.now(timezone.utc)
            await self.db.commit()
            return {
                "video_id": str(existing.id),
                "platform_id": platform_id,
                "status": "no_changes",
                "fields_changed": [],
                "trigger": trigger,
            }

        # Apply changes
        for field in changes["fields_changed"]:
            if hasattr(existing, field) and field in fresh_data:
                setattr(existing, field, fresh_data[field])

        existing.last_crawled_at = datetime.now(timezone.utc)
        existing.updated_at = datetime.now(timezone.utc)
        existing.status = VideoStatus.REINDEXING

        # Create recrawl event
        event = RecrawlEvent(
            video_id=existing.id,
            trigger=trigger,
            fields_changed=changes["fields_changed"],
            delta_json=changes["delta_json"],
            comments_added=changes.get("comments_added", 0),
            comments_deleted=changes.get("comments_deleted", 0),
            description_changed="description" in changes["fields_changed"],
            title_changed="title" in changes["fields_changed"],
        )
        self.db.add(event)

        await self.db.commit()
        await self.db.refresh(existing)

        return {
            "video_id": str(existing.id),
            "platform_id": platform_id,
            "status": "recrawled",
            "fields_changed": changes["fields_changed"],
            "delta_json": changes["delta_json"],
            "trigger": trigger,
            "recrawl_event_id": str(event.id),
        }

    async def check_and_recrawl_channel(
        self,
        platform_id: str,
        fresh_data: Dict[str, Any],
        trigger: str = "input_duplicate",
    ) -> Optional[Dict[str, Any]]:
        """Check if channel exists. If yes, diff and update."""
        result = await self.db.execute(
            select(Channel).where(Channel.platform_id == platform_id)
        )
        existing = result.scalar_one_or_none()

        if existing is None:
            return None

        changes = self._diff_channel(existing, fresh_data)
        if not changes["fields_changed"]:
            existing.last_crawled_at = datetime.now(timezone.utc)
            existing.updated_at = datetime.now(timezone.utc)
            await self.db.commit()
            return {
                "channel_id": str(existing.id),
                "status": "no_changes",
                "fields_changed": [],
            }

        for field in changes["fields_changed"]:
            if hasattr(existing, field) and field in fresh_data:
                setattr(existing, field, fresh_data[field])

        existing.last_crawled_at = datetime.now(timezone.utc)
        existing.updated_at = datetime.now(timezone.utc)
        await self.db.commit()

        return {
            "channel_id": str(existing.id),
            "status": "recrawled",
            "fields_changed": changes["fields_changed"],
            "delta_json": changes["delta_json"],
        }

    async def get_stale_videos(
        self,
        max_age_hours: int = 168,  # 7 days
        limit: int = 50,
    ) -> List[Dict]:
        """Find videos that haven't been crawled recently."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        result = await self.db.execute(
            select(Video)
            .where(
                (Video.last_crawled_at < cutoff) | (Video.last_crawled_at.is_(None))
            )
            .where(Video.status == VideoStatus.INDEXED)
            .order_by(Video.priority_score.desc())
            .limit(limit)
        )
        videos = result.scalars().all()

        return [
            {
                "video_id": str(v.id),
                "platform_id": v.platform_id,
                "title": v.title,
                "last_crawled_at": v.last_crawled_at.isoformat() if v.last_crawled_at else None,
                "updated_at": v.updated_at.isoformat() if v.updated_at else None,
                "hours_stale": int((datetime.now(timezone.utc) - (v.last_crawled_at or v.created_at)).total_seconds() / 3600),
                "priority_score": v.priority_score,
            }
            for v in videos
        ]

    async def get_stale_channels(
        self,
        max_age_hours: int = 336,  # 14 days
        limit: int = 20,
    ) -> List[Dict]:
        """Find channels that haven't been crawled recently."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)

        result = await self.db.execute(
            select(Channel)
            .where(
                (Channel.last_crawled_at < cutoff) | (Channel.last_crawled_at.is_(None))
            )
            .where(Channel.is_active == True)
            .order_by(Channel.priority_tier.desc())
            .limit(limit)
        )
        channels = result.scalars().all()

        return [
            {
                "channel_id": str(c.id),
                "platform_id": c.platform_id,
                "name": c.name,
                "last_crawled_at": c.last_crawled_at.isoformat() if c.last_crawled_at else None,
                "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                "priority_tier": c.priority_tier,
            }
            for c in channels
        ]

    async def get_recrawl_history(
        self,
        video_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """Get recrawl event history."""
        import uuid as uuid_mod
        query = select(RecrawlEvent).order_by(RecrawlEvent.created_at.desc()).limit(limit)
        if video_id:
            query = query.where(RecrawlEvent.video_id == uuid_mod.UUID(video_id))

        result = await self.db.execute(query)
        events = result.scalars().all()

        return [
            {
                "id": str(e.id),
                "video_id": str(e.video_id),
                "trigger": e.trigger,
                "fields_changed": e.fields_changed,
                "delta_json": e.delta_json,
                "comments_added": e.comments_added,
                "comments_deleted": e.comments_deleted,
                "description_changed": e.description_changed,
                "title_changed": e.title_changed,
                "duration_ms": e.duration_ms,
                "created_at": e.created_at.isoformat(),
            }
            for e in events
        ]

    # ── Private ──────────────────────────────────────────────────────

    def _diff_video(self, existing: Video, fresh: Dict) -> Dict:
        fields_changed = []
        delta = {}

        for field in VIDEO_DIFF_FIELDS:
            old_val = getattr(existing, field, None)
            new_val = fresh.get(field)
            if new_val is not None and old_val != new_val:
                fields_changed.append(field)
                delta[field] = [
                    _serialize(old_val),
                    _serialize(new_val),
                ]

        return {
            "fields_changed": fields_changed,
            "delta_json": delta,
        }

    def _diff_channel(self, existing: Channel, fresh: Dict) -> Dict:
        fields_changed = []
        delta = {}

        for field in CHANNEL_DIFF_FIELDS:
            old_val = getattr(existing, field, None)
            new_val = fresh.get(field)
            if new_val is not None and old_val != new_val:
                fields_changed.append(field)
                delta[field] = [
                    _serialize(old_val),
                    _serialize(new_val),
                ]

        return {
            "fields_changed": fields_changed,
            "delta_json": delta,
        }


def _serialize(val):
    """Make values JSON-safe."""
    if isinstance(val, datetime):
        return val.isoformat()
    if isinstance(val, (list, dict)):
        return val
    return val
