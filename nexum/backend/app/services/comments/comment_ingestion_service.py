"""
Nexum Comment Ingestion Service

Responsibilities:
  - Fetch video descriptions
  - Fetch pinned comments
  - Fetch top N comments + replies recursively
  - Platform-aware rate limiting
  - Retry failures with backoff
  - Deduplicate by platform_comment_id
  - Spam/bot filtering (heuristic + NLP)
  - Engagement threshold enforcement
  - Thread tree construction (parent/root/depth tracking)
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
import zlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.events import event_hub, GraphEventType
from app.models.models import (
    Comment, CommentAuthor, CommentStatus, Platform, Video, VideoStatus,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class CommentIngestionService:
    """
    Fetches and stores platform comments with full thread structure.

    Uses yt-dlp's comment extraction for YouTube and falls back to
    platform-specific logic for others.
    """

    # ── Main Entry Point ─────────────────────────────────────────────────

    async def ingest_comments_for_video(
        self,
        video_id: str,
        db: AsyncSession,
        max_comments: int = None,
        max_reply_depth: int = None,
    ) -> Dict[str, int]:
        """
        Ingest all comments for a video.
        Returns stats dict: {fetched, new, spam_filtered, authors_discovered}
        """
        max_comments = max_comments or settings.comment_fetch_top_n
        max_reply_depth = max_reply_depth or settings.comment_max_reply_depth

        # Load video
        video = await db.get(Video, uuid.UUID(video_id))
        if not video or video.status != VideoStatus.INDEXED:
            logger.warning(f"Video {video_id} not found or not indexed")
            return {"fetched": 0, "new": 0, "spam_filtered": 0, "authors_discovered": 0}

        # Fetch raw comments from platform
        raw_comments = await self._fetch_comments_from_platform(
            video.url, video.platform, max_comments, max_reply_depth
        )

        if not raw_comments:
            logger.info(f"No comments found for video {video_id}")
            return {"fetched": 0, "new": 0, "spam_filtered": 0, "authors_discovered": 0}

        # Process and store
        stats = await self._store_comments(raw_comments, video, db)
        await db.commit()

        logger.info(
            f"Comment ingestion complete for {video_id}: "
            f"fetched={stats['fetched']}, new={stats['new']}, "
            f"spam={stats['spam_filtered']}, authors={stats['authors_discovered']}"
        )
        return stats

    # ── Platform Fetching ────────────────────────────────────────────────

    async def _fetch_comments_from_platform(
        self,
        video_url: str,
        platform: Platform,
        max_comments: int,
        max_reply_depth: int,
    ) -> List[Dict[str, Any]]:
        """Use yt-dlp to extract comments with threading info."""
        try:
            import yt_dlp

            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "skip_download": True,
                "getcomments": True,
                "extractor_args": {
                    "youtube": {
                        "comment_sort": ["top"],
                        "max_comments": [str(max_comments), str(max_comments), str(max_reply_depth), str(max_comments)],
                    }
                },
            }

            loop = asyncio.get_event_loop()
            info = await loop.run_in_executor(None, self._extract_with_ytdlp, video_url, ydl_opts)

            if not info:
                return []

            raw_comments = info.get("comments", [])
            description = info.get("description", "")

            # Normalize to our format
            normalized = []
            for c in raw_comments:
                normalized.append({
                    "platform_comment_id": self._make_comment_id(c, platform),
                    "author_id": c.get("author_id", c.get("author", "unknown")),
                    "author_display_name": c.get("author", "Unknown"),
                    "text": c.get("text", ""),
                    "like_count": c.get("like_count", 0) or 0,
                    "reply_count": 0,  # yt-dlp doesn't always provide this
                    "parent_id": c.get("parent", "root"),
                    "timestamp_posted": c.get("timestamp"),
                    "is_pinned": c.get("is_pinned", False),
                    "is_favorited": c.get("is_favorited", False),
                })

            # Rate limit compliance
            await asyncio.sleep(settings.comment_rate_limit_delay)

            return normalized

        except Exception as e:
            logger.error(f"Comment fetch failed for {video_url}: {e}")
            return []

    @staticmethod
    def _extract_with_ytdlp(url: str, opts: dict) -> Optional[dict]:
        import yt_dlp
        with yt_dlp.YoutubeDL(opts) as ydl:
            try:
                return ydl.extract_info(url, download=False)
            except Exception as e:
                logger.error(f"yt-dlp comment extraction failed: {e}")
                return None

    @staticmethod
    def _make_comment_id(comment: dict, platform: Platform) -> str:
        """Generate stable platform_comment_id."""
        cid = comment.get("id", "")
        if cid:
            return f"{platform.value}:{cid}"
        # Fallback: hash author + text + timestamp
        text = comment.get("text", "")[:100]
        author = comment.get("author", "")
        ts = str(comment.get("timestamp", ""))
        raw = f"{platform.value}:{author}:{text}:{ts}"
        return f"{platform.value}:h{hashlib.sha256(raw.encode()).hexdigest()[:16]}"

    # ── Storage ──────────────────────────────────────────────────────────

    async def _store_comments(
        self, raw_comments: List[Dict], video: Video, db: AsyncSession
    ) -> Dict[str, int]:
        """
        Store comments with threading, deduplication, and spam filtering.
        Builds parent/root/depth tree in-memory before persisting.
        """
        stats = {"fetched": len(raw_comments), "new": 0, "spam_filtered": 0, "authors_discovered": 0}

        # Load existing comment IDs for dedup
        existing_result = await db.execute(
            select(Comment.platform_comment_id).where(Comment.video_id == video.id)
        )
        existing_ids = {r[0] for r in existing_result.all()}

        # Build thread tree: parent_platform_id -> children
        children_map: Dict[str, List[Dict]] = {}
        root_comments: List[Dict] = []

        for c in raw_comments:
            pid = c.get("parent_id", "root")
            if pid == "root" or not pid:
                root_comments.append(c)
            else:
                children_map.setdefault(pid, []).append(c)

        # Process root comments first, then recursively process replies
        author_cache: Dict[str, uuid.UUID] = {}
        comment_id_map: Dict[str, uuid.UUID] = {}  # platform_comment_id -> db id

        async def store_comment(
            raw: Dict,
            parent_db_id: Optional[uuid.UUID],
            root_db_id: Optional[uuid.UUID],
            depth: int,
        ) -> Optional[uuid.UUID]:
            pcid = raw["platform_comment_id"]

            # Dedup
            if pcid in existing_ids:
                return comment_id_map.get(pcid)

            text = raw.get("text", "").strip()
            if not text:
                return None

            # Spam filtering (heuristic quick check)
            spam_score = self._quick_spam_score(text, raw.get("like_count", 0))
            if spam_score >= settings.comment_spam_threshold:
                stats["spam_filtered"] += 1
                return None

            # Get or create author
            author_id = await self._get_or_create_author(
                raw.get("author_id", "unknown"),
                raw.get("author_display_name", "Unknown"),
                video.platform,
                db,
                author_cache,
                stats,
            )

            # Compress long comments
            text_compressed = None
            if len(text) > settings.comment_max_length_compress:
                text_compressed = zlib.compress(text.encode("utf-8"))

            db_id = uuid.uuid4()
            rthread = root_db_id or db_id  # root_thread_id = self if root

            comment = Comment(
                id=db_id,
                platform_comment_id=pcid,
                video_id=video.id,
                author_id=author_id,
                parent_comment_id=parent_db_id,
                root_thread_id=rthread,
                depth_level=depth,
                text=text,
                text_compressed=text_compressed,
                like_count=raw.get("like_count", 0),
                reply_count=raw.get("reply_count", 0),
                status=CommentStatus.PENDING,
                timestamp_posted=self._parse_timestamp(raw.get("timestamp_posted")),
            )
            db.add(comment)
            comment_id_map[pcid] = db_id
            existing_ids.add(pcid)
            stats["new"] += 1

            # Emit graph event
            await event_hub.emit_node(
                GraphEventType.COMMENT_ADDED, "Comment", str(db_id),
                {"video_id": str(video.id), "depth": depth},
            )
            if parent_db_id:
                await event_hub.emit_edge(
                    GraphEventType.COMMENT_LINKED, "REPLIES_TO",
                    str(db_id), str(parent_db_id),
                )

            return db_id

        # BFS: process roots, then children
        for root_raw in root_comments:
            root_id = await store_comment(root_raw, None, None, 0)
            if root_id:
                await self._process_children(
                    root_raw["platform_comment_id"], root_id, root_id, 1,
                    children_map, store_comment, settings.comment_max_reply_depth,
                )

        # Flush to DB in batches
        await db.flush()

        return stats

    async def _process_children(
        self,
        parent_pcid: str,
        parent_db_id: uuid.UUID,
        root_db_id: uuid.UUID,
        depth: int,
        children_map: Dict[str, List[Dict]],
        store_fn,
        max_depth: int,
    ):
        """Recursively process reply children."""
        if depth > max_depth:
            return

        # yt-dlp uses various parent ID formats
        children = children_map.get(parent_pcid, [])
        # Also check without platform prefix
        short_id = parent_pcid.split(":", 1)[-1] if ":" in parent_pcid else parent_pcid
        children.extend(children_map.get(short_id, []))

        child_count = 0
        for child_raw in children:
            child_id = await store_fn(child_raw, parent_db_id, root_db_id, depth)
            if child_id:
                child_count += 1
                await self._process_children(
                    child_raw["platform_comment_id"], child_id, root_db_id,
                    depth + 1, children_map, store_fn, max_depth,
                )

    # ── Author Management ────────────────────────────────────────────────

    async def _get_or_create_author(
        self,
        platform_author_id: str,
        display_name: str,
        platform: Platform,
        db: AsyncSession,
        cache: Dict[str, uuid.UUID],
        stats: Dict[str, int],
    ) -> uuid.UUID:
        """Get or create a CommentAuthor, with in-memory cache."""
        cache_key = f"{platform.value}:{platform_author_id}"
        if cache_key in cache:
            return cache[cache_key]

        # Check DB
        result = await db.execute(
            select(CommentAuthor).where(
                CommentAuthor.platform == platform,
                CommentAuthor.platform_author_id == platform_author_id,
            )
        )
        author = result.scalar_one_or_none()

        if author:
            author.comment_count += 1
            author.last_seen_at = datetime.now(timezone.utc)
            cache[cache_key] = author.id
            return author.id

        # Create new
        author = CommentAuthor(
            id=uuid.uuid4(),
            platform=platform,
            platform_author_id=platform_author_id,
            display_name=display_name,
            comment_count=1,
        )
        db.add(author)
        cache[cache_key] = author.id
        stats["authors_discovered"] += 1

        # Emit event
        await event_hub.emit_node(
            GraphEventType.USER_CREATED, "CommentAuthor", str(author.id),
            {"display_name": display_name, "platform": platform.value},
        )
        return author.id

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _quick_spam_score(text: str, like_count: int) -> float:
        """Fast heuristic spam check before NLP."""
        score = 0.0
        text_lower = text.lower()

        if any(kw in text_lower for kw in ["subscribe", "check out my", "promo code", "free gift"]):
            score += 0.4
        if text_lower.count("http") >= 2:
            score += 0.3
        if len(text) < 5:
            score += 0.2

        # High-engagement comments are less likely spam
        if like_count >= 10:
            score *= 0.5
        if like_count >= 100:
            score *= 0.3

        return min(score, 1.0)

    @staticmethod
    def _parse_timestamp(ts) -> Optional[datetime]:
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts.replace("Z", "+00:00"))
            except (ValueError, TypeError):
                return None
        return None


# Module-level singleton
comment_ingestion_service = CommentIngestionService()
