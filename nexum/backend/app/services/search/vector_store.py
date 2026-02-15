"""
Nexum Vector Store Service — manages Qdrant collections for text + visual embeddings.
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """Qdrant vector store with separate collections for text and visual."""

    def __init__(self):
        self.client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            timeout=30,
        )

    def ensure_collections(self):
        """Create collections if they don't exist."""
        for name, dim in [
            (settings.qdrant_collection_text, settings.text_embedding_dim),
            (settings.qdrant_collection_visual, settings.visual_embedding_dim),
            (settings.qdrant_collection_comments, settings.comment_embedding_dim),
            (settings.qdrant_collection_audio, settings.audio_embedding_dim),
        ]:
            if not self.client.collection_exists(name):
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=VectorParams(
                        size=dim,
                        distance=Distance.COSINE,
                    ),
                    optimizers_config=models.OptimizersConfigDiff(
                        indexing_threshold=20000,
                    ),
                    on_disk_payload=True,
                )
                # Create payload indices
                for field in ["video_id", "channel_id", "language", "model_version"]:
                    self.client.create_payload_index(
                        collection_name=name,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                self.client.create_payload_index(
                    collection_name=name,
                    field_name="timestamp",
                    field_schema=models.PayloadSchemaType.FLOAT,
                )
                logger.info(f"Created collection: {name} (dim={dim})")

        # Additional indices for comments collection
        if self.client.collection_exists(settings.qdrant_collection_comments):
            for field in ["comment_id", "author_id", "sentiment"]:
                try:
                    self.client.create_payload_index(
                        collection_name=settings.qdrant_collection_comments,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass  # Already exists

        # Additional indices for audio collection
        if self.client.collection_exists(settings.qdrant_collection_audio):
            for field in ["audio_segment_id", "dominant_source"]:
                try:
                    self.client.create_payload_index(
                        collection_name=settings.qdrant_collection_audio,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass
            for field in ["start_time", "music_probability", "bpm", "overall_manipulation"]:
                try:
                    self.client.create_payload_index(
                        collection_name=settings.qdrant_collection_audio,
                        field_name=field,
                        field_schema=models.PayloadSchemaType.FLOAT,
                    )
                except Exception:
                    pass

    # ── Upsert ───────────────────────────────────────────────────────────

    def upsert_text_embeddings(
        self,
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """Insert transcript segment embeddings."""
        return self._upsert(
            settings.qdrant_collection_text, embeddings, payloads, ids
        )

    def upsert_visual_embeddings(
        self,
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """Insert frame visual embeddings."""
        return self._upsert(
            settings.qdrant_collection_visual, embeddings, payloads, ids
        )

    def upsert_comment_embeddings(
        self,
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """Insert comment semantic embeddings."""
        return self._upsert(
            settings.qdrant_collection_comments, embeddings, payloads, ids
        )

    def upsert_audio_embeddings(
        self,
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        """Insert audio PANNs embeddings for acoustic search."""
        return self._upsert(
            settings.qdrant_collection_audio, embeddings, payloads, ids
        )

    def _upsert(
        self,
        collection: str,
        embeddings: List[np.ndarray],
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
    ) -> int:
        if not embeddings:
            return 0

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in embeddings]

        points = [
            PointStruct(
                id=pid,
                vector=emb.astype(np.float32).tolist(),
                payload=payload,
            )
            for pid, emb, payload in zip(ids, embeddings, payloads)
        ]

        batch_size = 256
        total = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection, points=batch)
            total += len(batch)

        return total

    # ── Search ───────────────────────────────────────────────────────────

    def search_text(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        return self._search(
            settings.qdrant_collection_text, query_vector, top_k, filters
        )

    def search_visual(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        return self._search(
            settings.qdrant_collection_visual, query_vector, top_k, filters
        )

    def search_comments(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search comment embeddings for semantic comment matching."""
        return self._search(
            settings.qdrant_collection_comments, query_vector, top_k, filters
        )

    def search_audio(
        self,
        query_vector: np.ndarray,
        top_k: int = 100,
        filters: Optional[Dict] = None,
    ) -> List[Dict]:
        """Search audio PANNs embeddings for acoustic event matching."""
        return self._search(
            settings.qdrant_collection_audio, query_vector, top_k, filters
        )

    def _search(
        self,
        collection: str,
        query_vector: np.ndarray,
        top_k: int,
        filters: Optional[Dict],
    ) -> List[Dict]:
        qdrant_filter = self._build_filter(filters) if filters else None

        results = self.client.search(
            collection_name=collection,
            query_vector=query_vector.astype(np.float32).tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=qdrant_filter,
        )

        return [
            {
                "id": str(r.id),
                "score": r.score,
                "payload": r.payload,
            }
            for r in results
        ]

    def _build_filter(self, filters: Dict) -> Filter:
        conditions = []
        if "video_id" in filters:
            conditions.append(
                FieldCondition(field_name="video_id", match=MatchValue(value=filters["video_id"]))
            )
        if "language" in filters:
            conditions.append(
                FieldCondition(field_name="language", match=MatchValue(value=filters["language"]))
            )
        if "channel_id" in filters:
            conditions.append(
                FieldCondition(field_name="channel_id", match=MatchValue(value=filters["channel_id"]))
            )
        if "min_confidence" in filters:
            conditions.append(
                FieldCondition(
                    field_name="confidence",
                    range=Range(gte=filters["min_confidence"]),
                )
            )
        return Filter(must=conditions) if conditions else None

    # ── Delete ───────────────────────────────────────────────────────────

    def delete_by_video(self, video_id: str):
        """Remove all vectors for a given video (both collections)."""
        for collection in [settings.qdrant_collection_text, settings.qdrant_collection_visual]:
            self.client.delete(
                collection_name=collection,
                points_selector=models.FilterSelector(
                    filter=Filter(
                        must=[FieldCondition(field_name="video_id", match=MatchValue(value=video_id))]
                    )
                ),
            )

    # ── Stats ────────────────────────────────────────────────────────────

    def get_collection_stats(self) -> Dict:
        stats = {}
        for name in [
            settings.qdrant_collection_text, settings.qdrant_collection_visual,
            settings.qdrant_collection_comments, settings.qdrant_collection_audio,
        ]:
            try:
                info = self.client.get_collection(name)
                stats[name] = {
                    "points_count": info.points_count,
                    "vectors_count": info.vectors_count,
                    "status": info.status.value,
                    "dimension": info.config.params.vectors.size if hasattr(info.config.params.vectors, "size") else "unknown",
                }
            except Exception:
                stats[name] = {"status": "not_found"}
        return stats

    def recreate_collections(self):
        """
        Drop and recreate all collections (for dimension upgrades like 768→1024).

        WARNING: This destroys all indexed data. Re-indexing required after.
        """
        for name in [
            settings.qdrant_collection_text, settings.qdrant_collection_visual,
            settings.qdrant_collection_comments, settings.qdrant_collection_audio,
        ]:
            try:
                if self.client.collection_exists(name):
                    self.client.delete_collection(name)
                    logger.warning(f"Deleted collection: {name}")
            except Exception as e:
                logger.error(f"Failed to delete collection {name}: {e}")

        self.ensure_collections()
        logger.info("All collections recreated with updated dimensions")


vector_store = VectorStore()
