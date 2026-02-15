"""
Nexum Audio Intelligence API Routes.

Endpoints for querying acoustic analysis data:
  - GET  /audio/{video_id}         — audio segments for a video
  - GET  /audio/{video_id}/summary — aggregated audio analysis summary
  - GET  /audio/stats/overview     — system-wide audio intelligence stats
  - GET  /audio/events/search      — search by acoustic event type
"""
from __future__ import annotations

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, distinct
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import get_db
from app.models.models import AudioSegment, Video
from app.schemas.schemas import (
    AudioSegmentSchema,
    AudioAnalysisSummary,
    AudioStatsSchema,
    ManipulationScoresSchema,
)

logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/audio", tags=["audio"])


@router.get("/{video_id}", response_model=List[AudioSegmentSchema])
async def get_audio_segments(
    video_id: str,
    source_filter: Optional[str] = Query(None, description="Filter by dominant source: music, speech, silence"),
    min_music_prob: Optional[float] = Query(None, ge=0, le=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Get audio intelligence segments for a video."""
    query = select(AudioSegment).where(
        AudioSegment.video_id == video_id
    )

    if source_filter:
        query = query.where(AudioSegment.dominant_source == source_filter)
    if min_music_prob is not None:
        query = query.where(AudioSegment.music_probability >= min_music_prob)

    query = query.order_by(AudioSegment.start_time)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    segments = result.scalars().all()

    return [
        AudioSegmentSchema(
            id=str(s.id),
            video_id=str(s.video_id),
            start_time=s.start_time,
            end_time=s.end_time,
            music_probability=s.music_probability,
            speech_probability=s.speech_probability,
            dominant_source=s.dominant_source,
            source_tags=s.source_tags,
            event_tags=s.event_tags,
            manipulation_scores=ManipulationScoresSchema(**(s.manipulation_scores or {})),
            overall_manipulation=s.overall_manipulation,
            bpm=s.bpm,
            musical_key=s.musical_key,
            loudness_lufs=s.loudness_lufs,
            dynamic_range_db=s.dynamic_range_db,
            spectral_centroid=s.spectral_centroid,
            harmonic_ratio=s.harmonic_ratio,
        )
        for s in segments
    ]


@router.get("/{video_id}/summary", response_model=AudioAnalysisSummary)
async def get_audio_summary(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get aggregated audio analysis summary for a video."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Aggregate audio segments
    result = await db.execute(
        select(
            func.count(AudioSegment.id).label("total"),
            func.sum(AudioSegment.end_time - AudioSegment.start_time).label("total_duration"),
            func.avg(AudioSegment.bpm).label("avg_bpm"),
            func.avg(AudioSegment.overall_manipulation).label("avg_manipulation"),
        ).where(AudioSegment.video_id == video_id)
    )
    row = result.first()

    # Source breakdown
    source_result = await db.execute(
        select(
            AudioSegment.dominant_source,
            func.sum(AudioSegment.end_time - AudioSegment.start_time).label("seconds"),
        )
        .where(AudioSegment.video_id == video_id)
        .group_by(AudioSegment.dominant_source)
    )
    source_breakdown = {r.dominant_source: float(r.seconds or 0) for r in source_result}

    # Most common key
    key_result = await db.execute(
        select(AudioSegment.musical_key, func.count().label("cnt"))
        .where(AudioSegment.video_id == video_id)
        .where(AudioSegment.musical_key.isnot(None))
        .group_by(AudioSegment.musical_key)
        .order_by(func.count().desc())
        .limit(1)
    )
    key_row = key_result.first()

    # Average manipulation
    manip_avg = None
    if row and row.avg_manipulation and row.avg_manipulation > 0.1:
        manip_avg = ManipulationScoresSchema(
            overall_manipulation=round(float(row.avg_manipulation or 0), 3)
        )

    return AudioAnalysisSummary(
        video_id=video_id,
        duration_seconds=float(row.total_duration or 0) if row else 0,
        total_windows=int(row.total or 0) if row else 0,
        global_bpm=round(float(row.avg_bpm), 1) if row and row.avg_bpm else None,
        global_key=key_row.musical_key if key_row else None,
        dominant_source=max(source_breakdown, key=source_breakdown.get) if source_breakdown else "unknown",
        total_music_seconds=source_breakdown.get("music", 0),
        total_speech_seconds=source_breakdown.get("speech", 0),
        total_silence_seconds=source_breakdown.get("silence", 0),
        manipulation_summary=manip_avg,
    )


@router.get("/events/search")
async def search_by_audio_event(
    event: str = Query(..., description="Acoustic event label to search: applause, laughter, music, etc."),
    min_confidence: float = Query(0.3, ge=0, le=1),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Search for video moments by acoustic event type."""
    sort_col = AudioSegment.music_probability.desc() if event.lower() == "music" else AudioSegment.start_time

    result = await db.execute(
        select(AudioSegment, Video.url, Video.title)
        .join(Video, AudioSegment.video_id == Video.id)
        .where(AudioSegment.event_tags.isnot(None))
        .order_by(sort_col)
        .limit(limit * 3)
    )

    matches = []
    for row in result:
        seg = row[0]
        url = row[1]
        title = row[2]

        tags = seg.event_tags or []
        for tag in tags:
            if (tag.get("label", "").lower() == event.lower()
                    and tag.get("confidence", 0) >= min_confidence):
                matches.append({
                    "video_id": str(seg.video_id),
                    "video_title": title,
                    "video_url": url,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "event": tag["label"],
                    "confidence": tag["confidence"],
                    "dominant_source": seg.dominant_source,
                    "bpm": seg.bpm,
                })
                break

        if seg.dominant_source and seg.dominant_source.lower() == event.lower():
            if not any(m["video_id"] == str(seg.video_id) and m["start_time"] == seg.start_time for m in matches):
                matches.append({
                    "video_id": str(seg.video_id),
                    "video_title": title,
                    "video_url": url,
                    "start_time": seg.start_time,
                    "end_time": seg.end_time,
                    "event": seg.dominant_source,
                    "confidence": seg.music_probability if event.lower() == "music" else seg.speech_probability,
                    "dominant_source": seg.dominant_source,
                    "bpm": seg.bpm,
                })

        if len(matches) >= limit:
            break

    return matches[:limit]


@router.get("/stats/overview", response_model=AudioStatsSchema)
async def get_audio_stats(db: AsyncSession = Depends(get_db)):
    """System-wide audio intelligence statistics."""
    total = await db.scalar(select(func.count(AudioSegment.id)))
    videos_with = await db.scalar(
        select(func.count(distinct(AudioSegment.video_id)))
    )
    avg_music = await db.scalar(
        select(func.avg(AudioSegment.music_probability))
    )
    avg_speech = await db.scalar(
        select(func.avg(AudioSegment.speech_probability))
    )
    manip_flagged = await db.scalar(
        select(func.count(AudioSegment.id)).where(
            AudioSegment.overall_manipulation > 0.4
        )
    )

    return AudioStatsSchema(
        total_audio_segments=total or 0,
        videos_with_audio_analysis=videos_with or 0,
        avg_music_ratio=round(float(avg_music or 0), 3),
        avg_speech_ratio=round(float(avg_speech or 0), 3),
        manipulation_flagged_count=manip_flagged or 0,
    )
