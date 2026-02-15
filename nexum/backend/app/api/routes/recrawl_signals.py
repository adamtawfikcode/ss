"""
Nexum API — Recrawl & Signals routes.

/recrawl/* — Stale content detection, recrawl history, manual triggers
/signals/* — Deep analytics signal data from demo generator
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query

recrawl_routes = APIRouter(prefix="/recrawl", tags=["recrawl"])
signal_routes = APIRouter(prefix="/signals", tags=["signals"])


# ═══════════════════════════════════════════════════════════════════════
# Recrawl Routes (work with real DB when available, demo fallback)
# ═══════════════════════════════════════════════════════════════════════

@recrawl_routes.get("/stale/videos")
async def get_stale_videos(
    max_age_hours: int = Query(168, ge=1, le=8760),
    limit: int = Query(50, ge=1, le=200),
):
    """Find videos that need recrawling based on staleness."""
    # In demo mode, generate synthetic stale data
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.videos:
        return {"stale_videos": [], "count": 0}

    import random
    stale = []
    for v in random.sample(gen.videos, k=min(limit, len(gen.videos))):
        hours = random.randint(max_age_hours, max_age_hours * 3)
        stale.append({
            "video_id": v["id"],
            "platform_id": v.get("platform_id", ""),
            "title": v.get("title", ""),
            "last_crawled_at": v.get("last_crawled_at"),
            "updated_at": v.get("updated_at"),
            "hours_stale": hours,
            "priority_score": round(random.uniform(0.1, 1.0), 3),
        })

    stale.sort(key=lambda x: x["hours_stale"], reverse=True)
    return {"stale_videos": stale, "count": len(stale)}


@recrawl_routes.get("/stale/channels")
async def get_stale_channels(
    max_age_hours: int = Query(336, ge=1, le=8760),
    limit: int = Query(20, ge=1, le=100),
):
    """Find channels that need recrawling."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.channels:
        return {"stale_channels": [], "count": 0}

    import random
    stale = []
    for ch in gen.channels:
        hours = random.randint(max_age_hours // 2, max_age_hours * 2)
        stale.append({
            "channel_id": ch["id"],
            "name": ch.get("name", ""),
            "last_crawled_at": ch.get("last_crawled_at"),
            "updated_at": ch.get("updated_at"),
            "hours_stale": hours,
            "priority_tier": ch.get("priority_tier", 1),
        })

    stale.sort(key=lambda x: x["hours_stale"], reverse=True)
    return {"stale_channels": stale[:limit], "count": len(stale[:limit])}


@recrawl_routes.get("/history")
async def get_recrawl_history(
    video_id: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """Get recrawl event history."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {"events": [], "count": 0}

    events = gen.signals.recrawl_events
    if video_id:
        events = [e for e in events if e.get("video_id") == video_id]

    events = sorted(events, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]
    return {"events": events, "count": len(events)}


@recrawl_routes.get("/stats")
async def get_recrawl_stats():
    """Summary statistics for recrawl activity."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {"total_recrawls": 0}

    events = gen.signals.recrawl_events
    triggers = {}
    for e in events:
        t = e.get("trigger", "unknown")
        triggers[t] = triggers.get(t, 0) + 1

    fields_changed_total = {}
    for e in events:
        for f in (e.get("fields_changed") or []):
            fields_changed_total[f] = fields_changed_total.get(f, 0) + 1

    return {
        "total_recrawls": len(events),
        "by_trigger": triggers,
        "fields_changed_frequency": fields_changed_total,
        "avg_comments_added": round(sum(e.get("comments_added", 0) for e in events) / max(len(events), 1), 1),
        "avg_comments_deleted": round(sum(e.get("comments_deleted", 0) for e in events) / max(len(events), 1), 1),
        "title_changes": sum(1 for e in events if e.get("title_changed")),
        "description_changes": sum(1 for e in events if e.get("description_changed")),
    }


# ═══════════════════════════════════════════════════════════════════════
# Signal Routes — Deep analytics data access
# ═══════════════════════════════════════════════════════════════════════

@signal_routes.get("/stats")
async def get_signal_stats():
    """Get counts for all signal types."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {"signals": {}, "total": 0}

    stats = gen.signals.stats()
    return {"signals": stats, "total": sum(stats.values())}


@signal_routes.get("/temporal-visual")
async def get_temporal_visual_signals(
    video_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Cursor heatmaps, text edits, face ratios, background changes."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    def _filter(items, vid_id):
        if vid_id:
            items = [i for i in items if i.get("video_id") == vid_id]
        return items[:limit]

    return {
        "cursor_heatmaps": _filter(s.cursor_heatmaps, video_id),
        "text_edit_events": _filter(s.text_edit_events, video_id),
        "face_camera_ratios": _filter(s.face_camera_ratios, video_id),
        "background_changes": s.background_changes[:limit],
    }


@signal_routes.get("/audio-micro")
async def get_audio_micro_signals(
    video_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Breathing patterns, ambient bleeds, room fingerprints, laughter events."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    def _filter(items, vid_id):
        if vid_id:
            items = [i for i in items if i.get("video_id") == vid_id]
        return items[:limit]

    return {
        "breathing_patterns": _filter(s.breathing_patterns, video_id),
        "ambient_bleeds": _filter(s.ambient_bleeds, video_id),
        "room_fingerprints": _filter(s.room_fingerprints, video_id),
        "laughter_events": _filter(s.laughter_events, video_id),
    }


@signal_routes.get("/behavioral")
async def get_behavioral_signals(
    video_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Upload patterns, title edits, thumbnails, link health."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    def _vfilter(items):
        if video_id:
            return [i for i in items if i.get("video_id") == video_id][:limit]
        return items[:limit]

    def _cfilter(items):
        if channel_id:
            return [i for i in items if i.get("channel_id") == channel_id][:limit]
        return items[:limit]

    upload_patterns = _cfilter(s.upload_patterns) if channel_id else _vfilter(s.upload_patterns)

    return {
        "upload_patterns": upload_patterns,
        "title_edit_history": _vfilter(s.title_edit_history),
        "thumbnail_analyses": _vfilter(s.thumbnail_analyses),
        "link_health": _vfilter(s.link_health),
    }


@signal_routes.get("/comment-archaeology")
async def get_comment_archaeology(
    video_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """Pinned comment history, sentiment drift, deleted shadows."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    def _filter(items):
        if video_id:
            return [i for i in items if i.get("video_id") == video_id][:limit]
        return items[:limit]

    return {
        "pinned_comment_history": _filter(s.pinned_comment_history),
        "sentiment_drifts": _filter(s.sentiment_drifts),
        "deleted_shadows": _filter(s.deleted_shadows),
    }


@signal_routes.get("/cross-video")
async def get_cross_video_forensics(
    video_id: Optional[str] = None,
    channel_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
):
    """B-roll reuse, sponsor segments, intro/outro evolution."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    def _vfilter(items):
        if video_id:
            return [i for i in items if i.get("video_id") == video_id][:limit]
        return items[:limit]

    def _cfilter(items):
        if channel_id:
            return [i for i in items if i.get("channel_id") == channel_id][:limit]
        return items[:limit]

    return {
        "broll_reuses": _vfilter(s.broll_reuses),
        "sponsor_segments": _vfilter(s.sponsor_segments),
        "intro_outro_evolutions": _cfilter(s.intro_outro_evolutions) if channel_id else _vfilter(s.intro_outro_evolutions),
    }


@signal_routes.get("/linguistic")
async def get_linguistic_signals(
    video_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Code-switching, filler words, hedging, vocabulary profiles."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    profiles = gen.signals.linguistic_profiles
    if video_id:
        profiles = [p for p in profiles if p.get("video_id") == video_id]

    return {"linguistic_profiles": profiles[:limit]}


@signal_routes.get("/graph-relational")
async def get_graph_relational_signals(
    channel_id: Optional[str] = None,
    author_id: Optional[str] = None,
    entity_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Commenter migrations, entity propagation, parasocial indices, topic gestation."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {}

    s = gen.signals

    migrations = s.commenter_migrations
    if author_id:
        migrations = [m for m in migrations if m.get("author_id") == author_id]
    if channel_id:
        migrations = [m for m in migrations if m.get("from_channel_id") == channel_id or m.get("to_channel_id") == channel_id]

    propagations = s.entity_propagations
    if entity_id:
        propagations = [p for p in propagations if p.get("entity_id") == entity_id]
    if channel_id:
        propagations = [p for p in propagations if p.get("origin_channel_id") == channel_id]

    parasocial = s.parasocial_indices
    if author_id:
        parasocial = [p for p in parasocial if p.get("author_id") == author_id]
    if channel_id:
        parasocial = [p for p in parasocial if p.get("channel_id") == channel_id]

    gestations = s.topic_gestations
    if channel_id:
        gestations = [g for g in gestations if g.get("channel_id") == channel_id]
    if entity_id:
        gestations = [g for g in gestations if g.get("entity_id") == entity_id]

    return {
        "commenter_migrations": migrations[:limit],
        "entity_propagations": propagations[:limit],
        "parasocial_indices": parasocial[:limit],
        "topic_gestations": gestations[:limit],
    }


@signal_routes.get("/video/{video_id}")
async def get_all_signals_for_video(video_id: str):
    """Get ALL signal data attached to a specific video — unified view."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {"video_id": video_id, "signals": {}}

    s = gen.signals

    def _vf(items):
        return [i for i in items if i.get("video_id") == video_id]

    # Also find the video node itself
    video = next((v for v in gen.videos if v["id"] == video_id), None)

    return {
        "video_id": video_id,
        "video": video,
        "signals": {
            "recrawl_events": _vf(s.recrawl_events),
            "cursor_heatmaps": _vf(s.cursor_heatmaps),
            "text_edit_events": _vf(s.text_edit_events),
            "face_camera_ratios": _vf(s.face_camera_ratios),
            "breathing_patterns": _vf(s.breathing_patterns),
            "ambient_bleeds": _vf(s.ambient_bleeds),
            "room_fingerprints": _vf(s.room_fingerprints),
            "laughter_events": _vf(s.laughter_events),
            "upload_patterns": _vf(s.upload_patterns),
            "title_edit_history": _vf(s.title_edit_history),
            "thumbnail_analyses": _vf(s.thumbnail_analyses),
            "link_health": _vf(s.link_health),
            "pinned_comment_history": _vf(s.pinned_comment_history),
            "sentiment_drifts": _vf(s.sentiment_drifts),
            "deleted_shadows": _vf(s.deleted_shadows),
            "broll_reuses": _vf(s.broll_reuses),
            "sponsor_segments": _vf(s.sponsor_segments),
            "intro_outro_evolutions": _vf(s.intro_outro_evolutions),
            "linguistic_profiles": _vf(s.linguistic_profiles),
        },
        "signal_counts": {
            k: len(v) for k, v in {
                "recrawl_events": _vf(s.recrawl_events),
                "cursor_heatmaps": _vf(s.cursor_heatmaps),
                "text_edit_events": _vf(s.text_edit_events),
                "face_camera_ratios": _vf(s.face_camera_ratios),
                "breathing_patterns": _vf(s.breathing_patterns),
                "ambient_bleeds": _vf(s.ambient_bleeds),
                "room_fingerprints": _vf(s.room_fingerprints),
                "laughter_events": _vf(s.laughter_events),
                "upload_patterns": _vf(s.upload_patterns),
                "title_edit_history": _vf(s.title_edit_history),
                "thumbnail_analyses": _vf(s.thumbnail_analyses),
                "link_health": _vf(s.link_health),
                "pinned_comment_history": _vf(s.pinned_comment_history),
                "sentiment_drifts": _vf(s.sentiment_drifts),
                "deleted_shadows": _vf(s.deleted_shadows),
                "broll_reuses": _vf(s.broll_reuses),
                "sponsor_segments": _vf(s.sponsor_segments),
                "intro_outro_evolutions": _vf(s.intro_outro_evolutions),
                "linguistic_profiles": _vf(s.linguistic_profiles),
            }.items()
        },
    }


@signal_routes.get("/channel/{channel_id}")
async def get_all_signals_for_channel(channel_id: str):
    """Get ALL signal data related to a specific channel."""
    from app.api.routes.demo import _get_generator
    gen = _get_generator()
    if not gen or not gen.signals:
        return {"channel_id": channel_id, "signals": {}}

    s = gen.signals
    channel = next((c for c in gen.channels if c["id"] == channel_id), None)
    channel_video_ids = {v["id"] for v in gen.videos if v.get("channel_id") == channel_id}

    def _cf(items):
        return [i for i in items if i.get("channel_id") == channel_id]

    def _cvf(items):
        return [i for i in items if i.get("video_id") in channel_video_ids]

    return {
        "channel_id": channel_id,
        "channel": channel,
        "video_count": len(channel_video_ids),
        "signals": {
            "background_changes": _cf(s.background_changes),
            "upload_patterns": _cf(s.upload_patterns) or _cvf(s.upload_patterns),
            "intro_outro_evolutions": _cf(s.intro_outro_evolutions),
            "thumbnail_analyses": _cvf(s.thumbnail_analyses),
            "room_fingerprints": _cvf(s.room_fingerprints),
            "commenter_migrations": [
                m for m in s.commenter_migrations
                if m.get("from_channel_id") == channel_id or m.get("to_channel_id") == channel_id
            ],
            "entity_propagations": [
                p for p in s.entity_propagations if p.get("origin_channel_id") == channel_id
            ],
            "parasocial_indices": _cf(s.parasocial_indices),
            "topic_gestations": _cf(s.topic_gestations),
        },
    }
