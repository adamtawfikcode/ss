"""
Nexum API — Feedback routes.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.schemas import FeedbackCreate, FeedbackResponse
from app.services.feedback.feedback_service import feedback_service

router = APIRouter(prefix="/feedback", tags=["Feedback"])


@router.post("", response_model=FeedbackResponse)
async def submit_feedback(
    data: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
):
    """Submit user feedback (upvote, downvote, timestamp correction, mismatch report)."""
    feedback_id = await feedback_service.submit_feedback(data, db)
    return FeedbackResponse(id=feedback_id, status="received")


@router.get("/stats")
async def get_feedback_stats(db: AsyncSession = Depends(get_db)):
    """Get aggregated feedback statistics."""
    return await feedback_service.get_feedback_stats(db)
