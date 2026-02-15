"""
Nexum v4 API Routes — Calibration, Alignment, Change Points.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.ml.calibration import calibration_service
from app.models.models import TranscriptAlignment, AcousticChangePoint
from app.schemas.schemas import (
    AlignmentSummarySchema, AlignmentResultSchema, AlignmentSignalSchema,
    CalibrationStatusSchema,
    ChangePointResultSchema, ChangePointSchema,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v4", tags=["v4-intelligence"])


# ═══════════════════════════════════════════════════════════════════════════
# Calibration
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/calibration/status", response_model=CalibrationStatusSchema)
async def get_calibration_status():
    """Get calibration status for all model targets."""
    return calibration_service.get_status()


@router.post("/calibration/fit")
async def trigger_calibration_fit():
    """Manually trigger calibration fitting from collected feedback."""
    results = calibration_service.fit_all()
    calibration_service.save()
    fitted = sum(1 for v in results.values() if v)
    return {
        "status": "completed",
        "targets_fitted": fitted,
        "total_targets": len(results),
        "details": results,
    }


@router.get("/calibration/test")
async def test_calibration(
    target: str = Query(..., description="e.g. audio.pitch_shift"),
    raw_score: float = Query(..., ge=0, le=1),
):
    """Test calibration for a specific target and raw score."""
    result = calibration_service.calibrate(target, raw_score)
    return result.to_dict()


# ═══════════════════════════════════════════════════════════════════════════
# Audio-Transcript Alignment
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/alignment/{video_id}", response_model=AlignmentSummarySchema)
async def get_alignment(
    video_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get audio-transcript alignment data for a video."""
    result = await db.execute(
        select(TranscriptAlignment)
        .where(TranscriptAlignment.video_id == video_id)
        .order_by(TranscriptAlignment.start_time)
    )
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(404, "No alignment data for this video")

    segments = []
    for r in rows:
        signals = []
        if r.signals_json:
            for s in r.signals_json:
                signals.append(AlignmentSignalSchema(
                    name=s.get("name", ""),
                    score=s.get("score", 0),
                    detail=s.get("detail", ""),
                ))
        segments.append(AlignmentResultSchema(
            start_time=r.start_time,
            end_time=r.end_time,
            alignment_score=r.alignment_score,
            quality_level=r.quality_level,
            signals=signals,
            warnings=r.warnings_json or [],
        ))

    overall = sum(s.alignment_score for s in segments) / max(len(segments), 1)
    total_warnings = sum(len(s.warnings) for s in segments)
    mismatch_regions = [
        [s.start_time, s.end_time]
        for s in segments
        if s.quality_level in ("poor", "mismatch")
    ]

    quality = "good" if overall >= 0.8 else "acceptable" if overall >= 0.55 else "poor" if overall >= 0.3 else "mismatch"

    return AlignmentSummarySchema(
        video_id=video_id,
        overall_score=round(overall, 3),
        overall_quality=quality,
        total_warnings=total_warnings,
        mismatch_regions=mismatch_regions,
        segments=segments,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Acoustic Change Points
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/change-points/{video_id}", response_model=ChangePointResultSchema)
async def get_change_points(
    video_id: str,
    min_magnitude: float = Query(0.0, ge=0, le=1),
    db: AsyncSession = Depends(get_db),
):
    """Get acoustic change points for a video."""
    query = (
        select(AcousticChangePoint)
        .where(AcousticChangePoint.video_id == video_id)
        .where(AcousticChangePoint.magnitude >= min_magnitude)
        .order_by(AcousticChangePoint.timestamp)
    )
    result = await db.execute(query)
    rows = result.scalars().all()

    if not rows:
        raise HTTPException(404, "No change points for this video")

    points = [
        ChangePointSchema(
            timestamp=r.timestamp,
            magnitude=r.magnitude,
            transition_type=r.transition_type,
            detail=r.detail or "",
            from_state=r.from_state or "",
            to_state=r.to_state or "",
        )
        for r in rows
    ]

    transitions = {}
    for p in points:
        transitions[p.transition_type] = transitions.get(p.transition_type, 0) + 1
    dominant = sorted(transitions, key=transitions.get, reverse=True)[:3]

    # Get video duration for scene count
    max_ts = max(p.timestamp for p in points) if points else 0

    return ChangePointResultSchema(
        video_id=video_id,
        total_change_points=len(points),
        total_duration=max_ts,
        num_scenes=len(points) + 1,
        dominant_transitions=dominant,
        change_points=points,
    )


@router.get("/change-points/search/transition")
async def search_by_transition(
    transition_type: str = Query(..., description="e.g. silence_to_music"),
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    """Search for change points by transition type."""
    result = await db.execute(
        select(AcousticChangePoint)
        .where(AcousticChangePoint.transition_type == transition_type)
        .order_by(AcousticChangePoint.magnitude.desc())
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {
            "video_id": str(r.video_id),
            "timestamp": r.timestamp,
            "magnitude": r.magnitude,
            "detail": r.detail,
        }
        for r in rows
    ]
