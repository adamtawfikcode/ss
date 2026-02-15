"""
Nexum Comment Processing Service

Orchestrates NLP processing for ingested comments:
  1. Load batch of PENDING comments from PostgreSQL
  2. Run CommentNLPService on each
  3. Update comment records with NLP outputs
  4. Upsert embeddings to Qdrant (nexum_comments collection)
  5. Sync nodes and edges to Neo4j graph
  6. Extract and store entities
  7. Mark comments as PROCESSED / SPAM / TOXIC
"""
from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.events import event_hub, GraphEventType
from app.ml.nlp.comment_nlp_service import comment_nlp_service, CommentNLPResult
from app.models.models import (
    Comment, CommentAuthor, CommentStatus, Entity, EntityMention, EntityType,
    Segment, Video,
)
from app.services.search.vector_store import vector_store

logger = logging.getLogger(__name__)
settings = get_settings()

# Entity type mapping from spaCy labels
SPACY_TO_ENTITY_TYPE = {
    "PERSON": EntityType.PERSON,
    "ORG": EntityType.ORGANIZATION,
    "GPE": EntityType.LOCATION,
    "LOC": EntityType.LOCATION,
    "PRODUCT": EntityType.PRODUCT,
    "EVENT": EntityType.EVENT,
    "WORK_OF_ART": EntityType.CONCEPT,
    "FAC": EntityType.LOCATION,
    "NORP": EntityType.ORGANIZATION,
    "LAW": EntityType.CONCEPT,
}


class CommentProcessingService:
    """Orchestrates NLP analysis and graph integration for comments."""

    async def process_pending_comments(
        self, db: AsyncSession, batch_size: int = None
    ) -> Dict[str, int]:
        """
        Process a batch of pending comments through the NLP pipeline.
        Returns stats: {processed, spam, toxic, entities_discovered, embeddings_upserted}
        """
        batch_size = batch_size or settings.comment_batch_size
        stats = {
            "processed": 0,
            "spam": 0,
            "toxic": 0,
            "entities_discovered": 0,
            "embeddings_upserted": 0,
        }

        # Fetch pending comments
        result = await db.execute(
            select(Comment)
            .where(Comment.status == CommentStatus.PENDING)
            .order_by(Comment.created_at)
            .limit(batch_size)
        )
        comments = list(result.scalars().all())

        if not comments:
            return stats

        logger.info(f"Processing {len(comments)} pending comments")

        # NLP batch processing
        texts = [c.text for c in comments]
        nlp_results = comment_nlp_service.process_batch(texts)

        # Process each result
        embeddings_to_upsert = []
        payloads_to_upsert = []
        embedding_ids = []

        for comment, nlp in zip(comments, nlp_results):
            # Update NLP fields
            comment.sentiment_score = nlp.sentiment_score
            comment.sentiment_label = nlp.sentiment_label
            comment.toxicity_score = nlp.toxicity_score
            comment.spam_score = nlp.spam_score
            comment.extracted_timestamps = nlp.extracted_timestamps
            comment.extracted_numbers = nlp.extracted_numbers
            comment.topic_labels = nlp.topic_labels
            comment.entities_json = [e["name"] for e in nlp.entities]
            comment.language = nlp.language or comment.language

            # Determine status
            if nlp.spam_score >= settings.comment_spam_threshold:
                comment.status = CommentStatus.SPAM
                stats["spam"] += 1
                continue
            if nlp.toxicity_score >= settings.comment_toxicity_threshold:
                comment.status = CommentStatus.TOXIC
                stats["toxic"] += 1
                continue

            comment.status = CommentStatus.PROCESSED
            stats["processed"] += 1

            # Prepare embedding for Qdrant
            if nlp.embedding is not None:
                eid = str(uuid.uuid4())
                comment.embedding_id = eid
                embeddings_to_upsert.append(nlp.embedding)
                embedding_ids.append(eid)
                payloads_to_upsert.append({
                    "video_id": str(comment.video_id),
                    "comment_id": str(comment.id),
                    "author_id": str(comment.author_id) if comment.author_id else None,
                    "text": comment.text[:500],
                    "sentiment": nlp.sentiment_label,
                    "sentiment_score": nlp.sentiment_score,
                    "depth_level": comment.depth_level,
                    "like_count": comment.like_count,
                    "timestamp_posted": comment.timestamp_posted.isoformat() if comment.timestamp_posted else None,
                    "topics": nlp.topic_labels,
                })

            # Extract entities to PG + Neo4j
            for ent_data in nlp.entities:
                await self._store_entity(
                    ent_data, comment, db, stats,
                )

            # Link comment to referenced timestamps (segments)
            if nlp.extracted_timestamp_seconds:
                await self._link_timestamp_references(
                    comment, nlp.extracted_timestamp_seconds, db,
                )

        # Batch upsert embeddings to Qdrant
        if embeddings_to_upsert:
            count = vector_store.upsert_comment_embeddings(
                embeddings_to_upsert, payloads_to_upsert, embedding_ids,
            )
            stats["embeddings_upserted"] = count

        await db.flush()
        logger.info(f"Comment processing complete: {stats}")
        return stats

    # ── Entity Storage ───────────────────────────────────────────────────

    async def _store_entity(
        self,
        ent_data: Dict,
        comment: Comment,
        db: AsyncSession,
        stats: Dict,
    ):
        """Get or create an Entity and record the mention."""
        canonical = ent_data["name"].strip().lower()
        if len(canonical) < 2:
            return

        ent_type = SPACY_TO_ENTITY_TYPE.get(ent_data.get("type", ""), EntityType.OTHER)

        # Check existing
        result = await db.execute(
            select(Entity).where(Entity.canonical_name == canonical)
        )
        entity = result.scalar_one_or_none()

        if not entity:
            entity = Entity(
                id=uuid.uuid4(),
                canonical_name=canonical,
                entity_type=ent_type,
                mention_count=1,
            )
            db.add(entity)
            stats["entities_discovered"] += 1

            await event_hub.emit_node(
                GraphEventType.ENTITY_DISCOVERED, "Entity", str(entity.id),
                {"name": canonical, "type": ent_type.value},
            )
        else:
            entity.mention_count += 1

        # Record mention
        mention = EntityMention(
            entity_id=entity.id,
            video_id=comment.video_id,
            source_type="comment",
            source_id=str(comment.id),
            context_text=comment.text[:200],
        )
        db.add(mention)

    # ── Timestamp Reference Linking ──────────────────────────────────────

    async def _link_timestamp_references(
        self,
        comment: Comment,
        timestamp_seconds: List[float],
        db: AsyncSession,
    ):
        """
        Link comment to video segments referenced by timestamps.
        e.g., "at 1:23 he says..." → REFERENCES_TIMESTAMP edge to the segment at 83s.
        """
        for ts in timestamp_seconds:
            result = await db.execute(
                select(Segment)
                .where(
                    Segment.video_id == comment.video_id,
                    Segment.start_time <= ts,
                    Segment.end_time >= ts,
                )
                .limit(1)
            )
            segment = result.scalar_one_or_none()
            if segment:
                await event_hub.emit_edge(
                    GraphEventType.COMMENT_LINKED, "REFERENCES_TIMESTAMP",
                    str(comment.id), str(segment.id),
                    {"timestamp": ts},
                )

    # ── Author Profile Aggregation ───────────────────────────────────────

    async def update_author_profiles(self, db: AsyncSession, limit: int = 100):
        """
        Periodically aggregate comment data into author profiles:
        topic_vector, sentiment_distribution, entity_frequency_map.
        """
        result = await db.execute(
            select(CommentAuthor)
            .order_by(CommentAuthor.last_seen_at.desc())
            .limit(limit)
        )
        authors = result.scalars().all()

        for author in authors:
            # Fetch their processed comments
            comments_result = await db.execute(
                select(Comment)
                .where(
                    Comment.author_id == author.id,
                    Comment.status == CommentStatus.PROCESSED,
                )
            )
            comments = comments_result.scalars().all()

            if not comments:
                continue

            # Aggregate sentiment distribution
            sentiments = {"positive": 0, "negative": 0, "neutral": 0}
            for c in comments:
                label = c.sentiment_label or "neutral"
                sentiments[label] = sentiments.get(label, 0) + 1
            author.sentiment_distribution = sentiments

            # Aggregate topic frequency
            topic_freq = {}
            for c in comments:
                for t in (c.topic_labels or []):
                    topic_freq[t] = topic_freq.get(t, 0) + 1
            author.entity_frequency_map = topic_freq

            # Update count
            author.comment_count = len(comments)

        await db.flush()


# Module-level singleton
comment_processing_service = CommentProcessingService()
