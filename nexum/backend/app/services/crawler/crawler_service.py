"""
Nexum Crawler Service — multi-platform video discovery and prioritization.

Supports:
  - YouTube channels/playlists
  - TikTok user pages
  - Twitter/X profiles
  - Any URL supported by yt-dlp (1700+ sites)

Responsibilities:
  - Read channels.txt for seed channels/accounts (multi-platform)
  - Discover new uploads via yt-dlp metadata extraction
  - Priority scoring for crawl ordering
  - Deduplication against existing DB entries
  - Rate limiting and retry logic
  - Insert processing jobs into Celery queue
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.models.models import Channel, CrawlLog, LiveStatus, Platform, Playlist, PlaylistItem, StreamInfo, Subtitle, Video, VideoStatus
from app.services.media.media_service import MediaProcessingService

logger = logging.getLogger(__name__)
settings = get_settings()


class CrawlerService:
    """Discovers and queues videos for processing from multiple platforms."""

    def __init__(self):
        self.rate_limit_delay = settings.crawler_rate_limit_delay
        self._priority_config = self._load_priority_config()

    # ── Platform Detection ───────────────────────────────────────────────

    @staticmethod
    def detect_platform(url: str) -> Platform:
        """Detect which platform a URL belongs to."""
        return MediaProcessingService.detect_platform(url)

    # ── Channel Management ───────────────────────────────────────────────

    def load_channels(self) -> List[Dict]:
        """Read channels/accounts from channels.txt (multi-platform)."""
        channels_file = Path(settings.channels_file)
        channels = []
        if not channels_file.exists():
            logger.warning(f"Channels file not found: {channels_file}")
            return channels

        for line in channels_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            url = parts[0].strip()
            priority = int(parts[1].strip()) if len(parts) > 1 else 1
            platform = self.detect_platform(url)
            channels.append({
                "url": url,
                "priority_tier": priority,
                "platform": platform,
            })

        logger.info(f"Loaded {len(channels)} channels/accounts from {channels_file}")
        return channels

    async def sync_channels_to_db(self, db: AsyncSession):
        """Ensure all channels from config are in the database."""
        channels = self.load_channels()
        for ch in channels:
            url = ch["url"]
            platform = ch["platform"]
            platform_id = self._extract_channel_id(url, platform)
            if not platform_id:
                continue

            existing = await db.execute(
                select(Channel).where(Channel.platform_id == platform_id)
            )
            if existing.scalar_one_or_none() is None:
                new_channel = Channel(
                    platform=platform,
                    platform_id=platform_id,
                    name=platform_id,  # updated later from metadata
                    url=url,
                    priority_tier=ch["priority_tier"],
                )
                db.add(new_channel)
                logger.info(f"Added channel [{platform.value}]: {platform_id}")
        await db.commit()

    # ── Video Discovery (Multi-Platform) ─────────────────────────────────

    async def discover_channel_videos(
        self, channel_url: str, platform: Platform, max_videos: int = 50
    ) -> List[Dict]:
        """Use yt-dlp to list recent videos from any supported platform."""
        import yt_dlp

        ydl_opts = {
            "extract_flat": True,
            "quiet": True,
            "no_warnings": True,
            "playlistend": max_videos,
            "ignoreerrors": True,
            "geo_bypass": True,
        }

        # Platform-specific discovery tweaks
        if platform == Platform.TIKTOK:
            ydl_opts["playlistend"] = min(max_videos, 30)  # TikTok rate limits
        elif platform == Platform.TWITTER:
            ydl_opts["playlistend"] = min(max_videos, 20)

        videos = []
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                if info and "entries" in info:
                    for entry in (info["entries"] or []):
                        if not entry or not entry.get("id"):
                            continue

                        # Build canonical URL per platform
                        video_url = self._build_video_url(entry, platform)

                        videos.append({
                            "platform_id": f"{platform.value}:{entry['id']}",
                            "platform": platform,
                            "title": entry.get("title", "Unknown"),
                            "url": video_url,
                            "duration": entry.get("duration"),
                            "view_count": entry.get("view_count"),
                            "upload_date": entry.get("upload_date"),
                            "thumbnail_url": entry.get("thumbnail") or entry.get("thumbnails", [{}])[0].get("url") if entry.get("thumbnails") else None,
                        })
                elif info and info.get("id"):
                    # Single video page (not a playlist)
                    video_url = self._build_video_url(info, platform)
                    videos.append({
                        "platform_id": f"{platform.value}:{info['id']}",
                        "platform": platform,
                        "title": info.get("title", "Unknown"),
                        "url": video_url,
                        "duration": info.get("duration"),
                        "view_count": info.get("view_count"),
                        "upload_date": info.get("upload_date"),
                        "thumbnail_url": info.get("thumbnail"),
                    })
        except Exception as e:
            logger.error(f"Failed to discover videos from {channel_url}: {e}")

        logger.info(f"Discovered {len(videos)} videos from [{platform.value}] {channel_url}")
        return videos

    @staticmethod
    def _build_video_url(entry: Dict, platform: Platform) -> str:
        """Build a canonical video URL from yt-dlp entry data."""
        vid_id = entry.get("id", "")
        orig_url = entry.get("url") or entry.get("webpage_url") or ""

        if platform == Platform.YOUTUBE:
            if "shorts" in orig_url:
                return f"https://www.youtube.com/shorts/{vid_id}"
            return f"https://www.youtube.com/watch?v={vid_id}"
        elif platform == Platform.TIKTOK:
            return orig_url or f"https://www.tiktok.com/video/{vid_id}"
        elif platform == Platform.TWITTER:
            return orig_url or f"https://twitter.com/i/status/{vid_id}"
        elif platform == Platform.INSTAGRAM:
            return orig_url or f"https://www.instagram.com/reel/{vid_id}"
        elif platform == Platform.TWITCH:
            return orig_url or f"https://www.twitch.tv/videos/{vid_id}"
        elif platform == Platform.VIMEO:
            return orig_url or f"https://vimeo.com/{vid_id}"
        else:
            return orig_url

    async def get_video_metadata(self, video_url: str) -> Optional[Dict]:
        """Fetch FULL metadata for a single video — every field yt-dlp exposes."""
        import yt_dlp

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["all"],
            "geo_bypass": True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                if not info:
                    return None

                platform = self.detect_platform(video_url)

                # Captions availability map
                manual_subs = list((info.get("subtitles") or {}).keys())
                auto_subs = list((info.get("automatic_captions") or {}).keys())
                captions_info = {"manual": manual_subs, "auto": auto_subs}

                # Chapters
                chapters = None
                if info.get("chapters"):
                    chapters = [
                        {"title": ch.get("title", ""), "start_time": ch.get("start_time", 0), "end_time": ch.get("end_time", 0)}
                        for ch in info["chapters"]
                    ]

                # Live status
                live_status = info.get("live_status") or ("was_live" if info.get("was_live") else "none")

                # Stream format info (keep top formats only)
                formats = []
                for fmt in (info.get("formats") or [])[-6:]:
                    entry = {"stream_type": "video" if fmt.get("vcodec", "none") != "none" else "audio"}
                    if fmt.get("vcodec") and fmt["vcodec"] != "none":
                        entry["codec"] = fmt["vcodec"]
                        entry["resolution"] = f"{fmt.get('width', '?')}x{fmt.get('height', '?')}"
                        entry["fps"] = fmt.get("fps")
                    else:
                        entry["codec"] = fmt.get("acodec")
                        entry["sample_rate"] = fmt.get("asr")
                        entry["channels"] = fmt.get("audio_channels")
                    entry["bitrate"] = fmt.get("tbr") or fmt.get("abr") or fmt.get("vbr")
                    entry["container_format"] = fmt.get("ext")
                    entry["file_size_bytes"] = fmt.get("filesize") or fmt.get("filesize_approx")
                    formats.append(entry)

                return {
                    "platform_id": f"{platform.value}:{info.get('id')}",
                    "platform": platform,
                    "title": info.get("title"),
                    "description": info.get("description"),
                    "url": video_url,
                    "thumbnail_url": info.get("thumbnail"),
                    "duration": info.get("duration"),
                    "view_count": info.get("view_count"),
                    "like_count": info.get("like_count"),
                    "comment_count": info.get("comment_count"),
                    "channel_name": info.get("channel") or info.get("uploader"),
                    "channel_id": info.get("channel_id") or info.get("uploader_id"),
                    "upload_date": info.get("upload_date"),
                    "language": info.get("language"),
                    "tags": info.get("tags") or [],
                    "categories": info.get("categories") or [],
                    "live_status": live_status,
                    "chapters": chapters,
                    "captions_info": captions_info,
                    "subtitles": info.get("subtitles"),
                    "automatic_captions": info.get("automatic_captions"),
                    "formats": formats,
                    "channel_follower_count": info.get("channel_follower_count"),
                    "uploader_url": info.get("uploader_url") or info.get("channel_url"),
                }
        except Exception as e:
            logger.error(f"Failed to get metadata for {video_url}: {e}")
        return None

    def compute_priority(self, video_meta: Dict, channel_priority: int = 1) -> float:
        """Compute priority score for crawl ordering."""
        cfg = self._priority_config

        views = video_meta.get("view_count") or 0
        duration = video_meta.get("duration") or 0
        upload_date_str = video_meta.get("upload_date")

        # View score (log-scaled)
        view_score = math.log10(max(views, 1)) * cfg.get("weight_views", 1.0)

        # Recency score
        recency = 0.0
        if upload_date_str:
            try:
                upload_dt = datetime.strptime(upload_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
                days_ago = (datetime.now(timezone.utc) - upload_dt).days
                recency = max(0, 100 - days_ago) * cfg.get("weight_recency", 0.5)
            except ValueError:
                pass

        # Duration check — wider range for TikTok (5s) to YouTube (4h)
        max_dur = settings.max_video_duration_seconds
        min_dur = settings.min_video_duration_seconds
        duration_penalty = 0.0
        if duration and (duration < min_dur or duration > max_dur):
            duration_penalty = 20.0
        elif duration and duration > 3600:
            duration_penalty = 5.0

        # Keyword boost
        keyword_boost = 0.0
        title = (video_meta.get("title") or "").lower()
        for kw, boost in cfg.get("keyword_boosts", {}).items():
            if kw.lower() in title:
                keyword_boost += boost

        # Engagement boost
        likes = video_meta.get("like_count") or 0
        engagement = math.log10(max(likes, 1)) * 0.5

        # Channel tier boost
        tier_multipliers = cfg.get("tier_multipliers", {})
        tier_boost = tier_multipliers.get(str(channel_priority), channel_priority) * 5

        score = view_score + recency - duration_penalty + keyword_boost + engagement + tier_boost
        return round(max(score, 0), 2)

    # ── Queue Management ─────────────────────────────────────────────────

    async def queue_video_for_processing(
        self, video_meta: Dict, priority: float, db: AsyncSession,
        channel_id: Optional[str] = None,
    ) -> Optional[str]:
        """Insert video into DB and queue Celery task."""
        pid = video_meta["platform_id"]

        # Dedup check
        existing = await db.execute(
            select(Video).where(Video.platform_id == pid)
        )
        if existing.scalar_one_or_none():
            return None

        # Parse upload date
        uploaded_at = None
        if video_meta.get("upload_date"):
            try:
                uploaded_at = datetime.strptime(
                    video_meta["upload_date"], "%Y%m%d"
                ).replace(tzinfo=timezone.utc)
            except ValueError:
                pass

        platform = video_meta.get("platform", Platform.OTHER)

        # Map live_status string to enum
        from app.models.models import LiveStatus
        live_status_map = {
            "is_live": LiveStatus.IS_LIVE,
            "was_live": LiveStatus.WAS_LIVE,
            "is_upcoming": LiveStatus.IS_UPCOMING,
            "post_live": LiveStatus.POST_LIVE,
        }
        live_status_val = live_status_map.get(video_meta.get("live_status"))

        video = Video(
            platform=platform,
            platform_id=pid,
            channel_id=channel_id,
            title=video_meta.get("title", "Unknown"),
            description=video_meta.get("description"),
            url=video_meta["url"],
            thumbnail_url=video_meta.get("thumbnail_url"),
            duration_seconds=video_meta.get("duration"),
            view_count=video_meta.get("view_count"),
            like_count=video_meta.get("like_count"),
            dislike_count=video_meta.get("dislike_count"),
            comment_count=video_meta.get("comment_count"),
            language=video_meta.get("language"),
            uploaded_at=uploaded_at,
            tags=video_meta.get("tags") or None,
            categories=video_meta.get("categories") or None,
            live_status=live_status_val,
            age_limit=video_meta.get("age_limit"),
            is_short=video_meta.get("is_short", False),
            has_captions=video_meta.get("has_captions", False),
            caption_languages=video_meta.get("caption_languages") or None,
            has_auto_captions=video_meta.get("has_auto_captions", False),
            auto_caption_languages=video_meta.get("auto_caption_languages") or None,
            status=VideoStatus.QUEUED,
            priority_score=priority,
        )
        db.add(video)
        await db.flush()

        video_id = str(video.id)

        # Queue Celery task
        from app.workers.tasks import process_video_task
        process_video_task.apply_async(
            args=[video_id],
            priority=min(int(priority), 9),
        )

        logger.info(f"Queued [{platform.value}]: {pid} ({video_meta.get('title')}) priority={priority}")
        return video_id

    # ── Crawl Loop ───────────────────────────────────────────────────────

    async def run_crawl_cycle(self):
        """One complete crawl cycle across all active channels."""
        async with async_session_factory() as db:
            await self.sync_channels_to_db(db)

            channels = await db.execute(
                select(Channel).where(Channel.is_active == True).order_by(Channel.priority_tier.desc())
            )
            channels = channels.scalars().all()

            total_queued = 0
            for channel in channels:
                try:
                    videos = await self.discover_channel_videos(
                        channel.url,
                        platform=channel.platform,
                        max_videos=30,
                    )
                    for v in videos:
                        priority = self.compute_priority(v, channel.priority_tier)
                        vid_id = await self.queue_video_for_processing(
                            v, priority, db, channel_id=str(channel.id)
                        )
                        if vid_id:
                            total_queued += 1

                    channel.last_crawled_at = datetime.now(timezone.utc)
                    await db.commit()

                    # Rate limit
                    await asyncio.sleep(self.rate_limit_delay)

                except Exception as e:
                    logger.error(f"Error crawling channel {channel.name}: {e}")
                    db.add(CrawlLog(
                        channel_id=channel.id,
                        platform=channel.platform,
                        action="crawl",
                        status="error",
                        details={"error": str(e)},
                    ))
                    await db.commit()

            logger.info(f"Crawl cycle complete. Queued {total_queued} new videos.")

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_channel_id(url: str, platform: Platform) -> Optional[str]:
        """Extract a unique channel identifier from a URL."""
        if platform == Platform.YOUTUBE:
            patterns = [
                r"youtube\.com/channel/([^/?]+)",
                r"youtube\.com/c/([^/?]+)",
                r"youtube\.com/@([^/?]+)",
                r"youtube\.com/user/([^/?]+)",
            ]
            for p in patterns:
                m = re.search(p, url)
                if m:
                    return f"yt:{m.group(1)}"
        elif platform == Platform.TIKTOK:
            m = re.search(r"tiktok\.com/@([^/?]+)", url)
            if m:
                return f"tt:{m.group(1)}"
        elif platform == Platform.TWITTER:
            m = re.search(r"(?:twitter|x)\.com/([^/?]+)", url)
            if m and m.group(1) not in ("i", "home", "search", "explore"):
                return f"tw:{m.group(1)}"
        elif platform == Platform.INSTAGRAM:
            m = re.search(r"instagram\.com/([^/?]+)", url)
            if m and m.group(1) not in ("p", "reel", "stories", "explore"):
                return f"ig:{m.group(1)}"
        elif platform == Platform.TWITCH:
            m = re.search(r"twitch\.tv/([^/?]+)", url)
            if m and m.group(1) not in ("directory", "videos"):
                return f"twitch:{m.group(1)}"

        # Fallback: hash the URL
        slug = url.rstrip("/").split("/")[-1]
        return f"{platform.value}:{slug}" if slug else None

    @staticmethod
    def _load_priority_config() -> Dict:
        config_path = Path(settings.config_dir) / "priority_rules.yaml"
        if config_path.exists():
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        return {
            "weight_views": 1.0,
            "weight_recency": 0.5,
            "keyword_boosts": {},
            "min_views": 0,
        }


crawler_service = CrawlerService()
