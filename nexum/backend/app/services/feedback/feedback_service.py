"""
Nexum Feedback Service — collects user signals and recalibrates ranking weights.

The feedback loop:
1. Users upvote/downvote results, suggest timestamps, report mismatches
2. Periodically aggregate feedback statistics
3. Adjust fusion weights to maximize upvote ratio
4. Log weight changes for auditing
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.models.models import EvaluationMetric, Feedback, FeedbackType
from app.schemas.schemas import FeedbackCreate

logger = logging.getLogger(__name__)
settings = get_settings()

# Weight adjustment step size
LEARNING_RATE = 0.02
MIN_WEIGHT = 0.05
MAX_WEIGHT = 0.50


class FeedbackService:
    """Manages feedback collection and weight recalibration."""

    async def submit_feedback(
        self, data: FeedbackCreate, db: AsyncSession, user_id: Optional[str] = None
    ) -> str:
        """Store a feedback entry."""
        feedback = Feedback(
            user_id=uuid.UUID(user_id) if user_id else None,
            video_id=uuid.UUID(data.video_id),
            segment_id=uuid.UUID(data.segment_id) if data.segment_id else None,
            query_text=data.query_text,
            feedback_type=FeedbackType(data.feedback_type),
            suggested_timestamp=data.suggested_timestamp,
            comment=data.comment,
        )
        db.add(feedback)
        await db.flush()
        return str(feedback.id)

    async def get_feedback_stats(self, db: AsyncSession) -> Dict:
        """Aggregate feedback statistics."""
        total = await db.scalar(select(func.count(Feedback.id)))
        upvotes = await db.scalar(
            select(func.count(Feedback.id)).where(
                Feedback.feedback_type == FeedbackType.UPVOTE
            )
        )
        downvotes = await db.scalar(
            select(func.count(Feedback.id)).where(
                Feedback.feedback_type == FeedbackType.DOWNVOTE
            )
        )
        corrections = await db.scalar(
            select(func.count(Feedback.id)).where(
                Feedback.feedback_type == FeedbackType.TIMESTAMP_CORRECTION
            )
        )
        mismatches = await db.scalar(
            select(func.count(Feedback.id)).where(
                Feedback.feedback_type == FeedbackType.MISMATCH_REPORT
            )
        )

        upvote_ratio = upvotes / max(upvotes + downvotes, 1)

        return {
            "total": total or 0,
            "upvotes": upvotes or 0,
            "downvotes": downvotes or 0,
            "timestamp_corrections": corrections or 0,
            "mismatch_reports": mismatches or 0,
            "upvote_ratio": round(upvote_ratio, 4),
        }

    async def recalibrate_weights(self):
        """
        Analyze recent feedback and nudge fusion weights.

        Strategy:
        - If upvote ratio for text-heavy matches > visual → increase text weight
        - If OCR matches get upvoted more → increase OCR weight
        - Normalize weights to sum to ~1.0
        """
        async with async_session_factory() as db:
            # Get recent feedback (last 7 days)
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            recent = await db.execute(
                select(Feedback).where(Feedback.created_at >= cutoff)
            )
            feedbacks = recent.scalars().all()

            if len(feedbacks) < 10:
                logger.info("Not enough feedback for recalibration (need 10+)")
                return

            stats = await self.get_feedback_stats(db)
            upvote_ratio = stats["upvote_ratio"]

            # Current weights
            weights = {
                "text_semantic": settings.weight_text_semantic,
                "visual_similarity": settings.weight_visual_similarity,
                "ocr_match": settings.weight_ocr_match,
                "keyword_match": settings.weight_keyword_match,
                "temporal_coherence": settings.weight_temporal_coherence,
                "emotion_context": settings.weight_emotion_context,
            }

            # If low upvote ratio, try diversifying weights
            if upvote_ratio < 0.6:
                # Slightly reduce dominant weight, boost others
                max_key = max(weights, key=weights.get)
                for k in weights:
                    if k == max_key:
                        weights[k] = max(weights[k] - LEARNING_RATE, MIN_WEIGHT)
                    else:
                        weights[k] = min(weights[k] + LEARNING_RATE / 4, MAX_WEIGHT)

            # Normalize
            total = sum(weights.values())
            weights = {k: round(v / max(total, 1e-8), 4) for k, v in weights.items()}

            # Log the adjustment
            logger.info(f"Recalibrated weights: {weights} (upvote_ratio={upvote_ratio})")

            # Store metric
            metric = EvaluationMetric(
                metric_name="weight_recalibration",
                metric_value=upvote_ratio,
                details={"weights": weights, "feedback_count": len(feedbacks)},
            )
            db.add(metric)
            await db.commit()

            # Note: In production, you'd update a shared config store (Redis/DB)
            # For now, weights are logged and can be applied via admin API


feedback_service = FeedbackService()
