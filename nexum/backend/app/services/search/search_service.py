"""
Nexum Search Service — multimodal fusion search engine.

Pipeline:
1. Query understanding (decompose into dimensions)
2. Vector search across text + visual collections
3. Keyword + OCR matching
4. Multimodal score fusion
5. Temporal coherence boosting
6. Re-ranking (top K)
7. Explanation generation
"""
from __future__ import annotations

import logging
import re
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.ml.embeddings.embedding_service import embedding_service
from app.models.models import Frame, Segment, Video, Channel, TranscriptAlignment, AcousticChangePoint
from app.schemas.schemas import (
    ModalityScore,
    QueryDecomposition as QueryDecompSchema,
    SearchFilters,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from app.services.search.query_understanding import query_pipeline
from app.services.search.vector_store import vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchService:
    """Multimodal fusion search across text, visual, and OCR modalities."""

    async def search(
        self, request: SearchRequest, db: AsyncSession
    ) -> SearchResponse:
        start = time.time()

        # 1. Query Understanding
        decomp = query_pipeline.decompose(request.query)
        sub_queries = query_pipeline.build_search_queries(decomp)

        # 2. Generate query embeddings
        text_query_vec = embedding_service.embed_text(request.query, is_query=True)
        clip_query_vec = embedding_service.embed_text_clip(request.query)

        # 3. Build vector store filters
        vs_filters = self._build_vs_filters(request.filters)

        # 4. Vector search — text modality
        text_results = vector_store.search_text(
            text_query_vec, top_k=settings.search_top_k, filters=vs_filters
        )

        # 5. Vector search — visual modality
        visual_results = vector_store.search_visual(
            clip_query_vec, top_k=settings.search_top_k, filters=vs_filters
        )

        # 5b. Vector search — comment modality (v3)
        comment_results = vector_store.search_comments(
            text_query_vec, top_k=settings.search_top_k, filters=vs_filters
        )

        # 5c. Vector search — audio modality (v3.1)
        audio_results = self._fetch_audio_context(vs_filters)

        # 5d. Fetch alignment + change point data (v4) — scoped to search results
        candidate_video_ids = set()
        for r in text_results:
            candidate_video_ids.add(r.payload.get("video_id", ""))
        for r in visual_results:
            candidate_video_ids.add(r.payload.get("video_id", ""))
        candidate_video_ids.discard("")

        alignment_data = await self._fetch_alignment_data(db, candidate_video_ids)
        change_point_data = await self._fetch_change_point_data(db, candidate_video_ids)

        # 6. Fuse results (14 modalities)
        fused = self._fuse_results(
            text_results=text_results,
            visual_results=visual_results,
            decomp=decomp,
            comment_results=comment_results,
            audio_results=audio_results,
            alignment_data=alignment_data,
            change_point_data=change_point_data,
        )

        # 7. Temporal coherence boost
        fused = self._apply_temporal_coherence(fused)

        # 8. Sort by final score
        fused.sort(key=lambda x: -x["final_score"])

        # 9. Paginate
        offset = (request.page - 1) * request.page_size
        page_results = fused[offset : offset + request.page_size]

        # 10. Hydrate from DB
        search_results = await self._hydrate_results(page_results, db, decomp)

        latency = (time.time() - start) * 1000

        return SearchResponse(
            query=request.query,
            total_results=len(fused),
            page=request.page,
            page_size=request.page_size,
            results=search_results,
            query_decomposition=QueryDecompSchema(
                objects=decomp.objects,
                actions=decomp.actions,
                numbers=decomp.numbers,
                time_era=decomp.time_era,
                emotion=decomp.emotion,
                context=decomp.context,
            ),
            latency_ms=round(latency, 1),
        )

    # ── Fusion ───────────────────────────────────────────────────────────

    def _fuse_results(
        self,
        text_results: List[Dict],
        visual_results: List[Dict],
        decomp,
        comment_results: Optional[List[Dict]] = None,
        entity_matches: Optional[Dict[str, float]] = None,
        audio_results: Optional[Dict[str, List[Dict]]] = None,
        alignment_data: Optional[Dict[str, List[Dict]]] = None,
        change_point_data: Optional[Dict[str, List[float]]] = None,
    ) -> List[Dict]:
        """
        Merge text, visual, comment, and audio hits by (video_id, timestamp) key.
        Compute composite score using 15-modality configurable weights.
        """
        w = {
            "text_semantic": settings.weight_text_semantic,
            "visual_similarity": settings.weight_visual_similarity,
            "ocr_match": settings.weight_ocr_match,
            "keyword_match": settings.weight_keyword_match,
            "temporal_coherence": settings.weight_temporal_coherence,
            "emotion_context": settings.weight_emotion_context,
            "comment_semantic": settings.weight_comment_semantic,
            "comment_timestamp_boost": settings.weight_comment_timestamp_boost,
            "agreement_cluster": settings.weight_agreement_cluster,
            "entity_overlap": settings.weight_entity_overlap,
            "user_cluster_confidence": settings.weight_user_cluster_confidence,
            "audio_event_match": settings.weight_audio_event_match,
            "audio_attribute_match": settings.weight_audio_attribute_match,
            # v4: new modalities
            "alignment_quality": settings.weight_alignment_quality,
            "change_point_proximity": settings.weight_change_point_proximity,
        }

        # Index by (video_id, timestamp_bucket)
        merged: Dict[str, Dict] = {}

        for r in text_results:
            key = self._result_key(r)
            entry = merged.setdefault(key, self._empty_entry(r))
            entry["text_semantic"] = r["score"]
            entry["transcript"] = r["payload"].get("text", "")

            # Keyword match
            if decomp.keywords:
                text_lower = entry["transcript"].lower()
                hits = sum(1 for kw in decomp.keywords if kw in text_lower)
                entry["keyword_match"] = min(hits / max(len(decomp.keywords), 1), 1.0)

            # OCR match from payload
            ocr = r["payload"].get("ocr_text", "")
            if ocr and decomp.numbers:
                ocr_hits = sum(1 for n in decomp.numbers if n in ocr)
                entry["ocr_match"] = min(ocr_hits / max(len(decomp.numbers), 1), 1.0)
                entry["ocr_text"] = ocr

            entry["confidence"] = r["payload"].get("confidence", 0.5)

        for r in visual_results:
            key = self._result_key(r)
            entry = merged.setdefault(key, self._empty_entry(r))
            entry["visual_similarity"] = r["score"]
            entry["visual_tags"] = r["payload"].get("visual_tags", [])

        # ── Comment Modality (v3) ────────────────────────────────────────
        if comment_results:
            comment_by_video: Dict[str, List[Dict]] = {}
            for cr in comment_results:
                vid = cr["payload"].get("video_id", "")
                comment_by_video.setdefault(vid, []).append(cr)

            for key, entry in merged.items():
                vid = entry["video_id"]
                if vid in comment_by_video:
                    # Best comment match score for this video
                    best_comment = max(comment_by_video[vid], key=lambda x: x["score"])
                    entry["comment_semantic"] = best_comment["score"]

                    # Comment timestamp boost: if a comment references the same timestamp
                    comment_ts = best_comment["payload"].get("extracted_timestamps", [])
                    if comment_ts and entry["timestamp"] > 0:
                        for cts in comment_ts:
                            try:
                                if abs(float(cts) - entry["timestamp"]) < 30:
                                    entry["comment_timestamp_boost"] = 0.8
                                    break
                            except (ValueError, TypeError):
                                pass

                    # Agreement cluster: sentiment consensus
                    sentiments = [c["payload"].get("sentiment_score", 0) for c in comment_by_video[vid]]
                    if len(sentiments) >= 3:
                        import statistics
                        try:
                            variance = statistics.variance(sentiments)
                            entry["agreement_cluster"] = max(0, 1.0 - variance * 2)
                        except statistics.StatisticsError:
                            pass

                    # User cluster confidence: many different users = higher signal
                    unique_authors = set(c["payload"].get("author_id") for c in comment_by_video[vid] if c["payload"].get("author_id"))
                    if len(unique_authors) >= 5:
                        entry["user_cluster_confidence"] = min(len(unique_authors) / 20, 1.0)

        # ── Entity Overlap (v3) ──────────────────────────────────────────
        if entity_matches:
            for key, entry in merged.items():
                vid = entry["video_id"]
                entry["entity_overlap"] = entity_matches.get(vid, 0.0)

        # ── Audio Modalities (v3.1) ─────────────────────────────────────
        if audio_results:
            # audio_results is Dict[video_id, List[audio_segment_info]]
            audio_keywords = set()
            for kw in (decomp.objects or []) + (decomp.actions or []):
                audio_keywords.add(kw.lower())

            # Known acoustic terms for matching
            acoustic_terms = {
                "music", "singing", "applause", "laughter", "cheering", "crowd",
                "gunshot", "explosion", "siren", "silence", "noise", "typing",
                "fast", "slow", "upbeat", "dramatic", "bass", "loud", "quiet",
                "robotic", "autotune", "echo", "reverb", "pitch", "speed",
            }

            for key, entry in merged.items():
                vid = entry["video_id"]
                ts = entry["timestamp"]

                if vid not in audio_results:
                    continue

                audio_segs = audio_results[vid]

                # Find audio segments overlapping this timestamp
                matching = [
                    a for a in audio_segs
                    if a.get("start_time", 0) <= ts + 10 and a.get("end_time", 0) >= ts - 5
                ]
                if not matching:
                    matching = audio_segs[:3]  # fall back to first few

                # Audio Event Match: do query keywords match detected events?
                event_score = 0.0
                for seg in matching:
                    seg_events = set(seg.get("event_labels", []))
                    seg_events.add(seg.get("dominant_source", ""))
                    overlap = audio_keywords & seg_events
                    if overlap:
                        event_score = max(event_score, len(overlap) / max(len(audio_keywords & acoustic_terms), 1))
                entry["audio_event_match"] = min(event_score, 1.0)

                # Audio Attribute Match: tempo/loudness/key matching
                attr_score = 0.0
                query_lower = decomp.raw_query.lower() if hasattr(decomp, "raw_query") else ""
                for seg in matching:
                    subscore = 0.0
                    bpm = seg.get("bpm")

                    # Tempo matching
                    if bpm:
                        if ("fast" in query_lower or "upbeat" in query_lower) and bpm > 120:
                            subscore += 0.4
                        elif ("slow" in query_lower or "dramatic" in query_lower) and bpm < 80:
                            subscore += 0.4

                    # Loudness matching
                    loudness = seg.get("loudness_lufs", -30)
                    if ("loud" in query_lower or "bass" in query_lower) and loudness > -15:
                        subscore += 0.3
                    elif ("quiet" in query_lower or "soft" in query_lower) and loudness < -25:
                        subscore += 0.3

                    # Harmonic content matching
                    hr = seg.get("harmonic_ratio", 0.5)
                    if "music" in query_lower and hr > 0.6:
                        subscore += 0.3

                    # Manipulation query matching
                    manip = seg.get("overall_manipulation", 0)
                    if any(t in query_lower for t in ["robotic", "autotune", "speed", "pitch", "filter"]):
                        if manip > 0.3:
                            subscore += 0.4

                    attr_score = max(attr_score, subscore)

                entry["audio_attribute_match"] = min(attr_score, 1.0)

        # ── Alignment Quality (v4) ──────────────────────────────────────
        if alignment_data:
            for key, entry in merged.items():
                vid = entry["video_id"]
                ts = entry["timestamp"]
                if vid in alignment_data:
                    # Find alignment segment closest to this timestamp
                    best_score = 0.0
                    for aln in alignment_data[vid]:
                        if aln.get("start_time", 0) <= ts + 15 and aln.get("end_time", 0) >= ts - 5:
                            best_score = max(best_score, aln.get("alignment_score", 0))
                    entry["alignment_quality"] = best_score

        # ── Change Point Proximity (v4) ──────────────────────────────────
        if change_point_data:
            for key, entry in merged.items():
                vid = entry["video_id"]
                ts = entry["timestamp"]
                if vid in change_point_data:
                    # Boost results near acoustic change points (interesting moments)
                    min_dist = float("inf")
                    for cp_ts in change_point_data[vid]:
                        dist = abs(ts - cp_ts)
                        min_dist = min(min_dist, dist)
                    # Closer to change point = higher score (10s window)
                    if min_dist < 10:
                        entry["change_point_proximity"] = max(0, 1.0 - min_dist / 10.0)

        # Compute final score with all 14 modalities
        for key, entry in merged.items():
            score = sum(
                entry.get(modality, 0) * weight
                for modality, weight in w.items()
            )
            # Confidence weighting
            conf = entry.get("confidence", 0.5)
            score *= (0.7 + 0.3 * conf)

            entry["final_score"] = score

        return list(merged.values())

    def _apply_temporal_coherence(self, results: List[Dict]) -> List[Dict]:
        """Boost results whose neighbors (same video, ±1 segment) also match."""
        by_video = defaultdict(list)
        for r in results:
            by_video[r["video_id"]].append(r)

        for vid, entries in by_video.items():
            entries.sort(key=lambda x: x["timestamp"])
            for i, e in enumerate(entries):
                neighbors = 0
                if i > 0 and abs(entries[i - 1]["timestamp"] - e["timestamp"]) < 30:
                    neighbors += 1
                if i < len(entries) - 1 and abs(entries[i + 1]["timestamp"] - e["timestamp"]) < 30:
                    neighbors += 1
                if neighbors > 0:
                    e["temporal_coherence"] = min(neighbors * 0.15, 0.3)
                    e["final_score"] += e["temporal_coherence"] * settings.weight_temporal_coherence

        return results

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _result_key(r: Dict) -> str:
        vid = r["payload"].get("video_id", "")
        ts = int(r["payload"].get("timestamp", 0) // 10) * 10  # 10s buckets
        return f"{vid}:{ts}"

    @staticmethod
    def _empty_entry(r: Dict) -> Dict:
        return {
            "video_id": r["payload"].get("video_id", ""),
            "timestamp": r["payload"].get("timestamp", 0.0),
            "end_timestamp": r["payload"].get("end_time", None),
            "text_semantic": 0.0,
            "visual_similarity": 0.0,
            "ocr_match": 0.0,
            "keyword_match": 0.0,
            "temporal_coherence": 0.0,
            "emotion_context": 0.0,
            "comment_semantic": 0.0,
            "comment_timestamp_boost": 0.0,
            "agreement_cluster": 0.0,
            "entity_overlap": 0.0,
            "user_cluster_confidence": 0.0,
            "audio_event_match": 0.0,
            "audio_attribute_match": 0.0,
            "alignment_quality": 0.0,
            "change_point_proximity": 0.0,
            "final_score": 0.0,
            "transcript": "",
            "visual_tags": [],
            "ocr_text": "",
            "confidence": 0.5,
        }

    def _fetch_audio_context(self, vs_filters: Optional[Dict] = None) -> Dict[str, List[Dict]]:
        """
        Fetch audio segment metadata grouped by video_id for fusion scoring.

        Unlike text/visual search which uses embedding similarity, audio modality
        scoring is attribute-based (event labels, BPM, loudness, manipulation scores).
        We fetch audio payloads from Qdrant using scroll (no vector query needed).
        """
        try:
            # Scroll through audio collection to get payloads
            records, _ = vector_store.client.scroll(
                collection_name=settings.qdrant_collection_audio,
                limit=2000,
                with_payload=True,
                with_vectors=False,
            )

            audio_by_video: Dict[str, List[Dict]] = {}
            for record in records:
                payload = record.payload or {}
                vid = payload.get("video_id", "")
                if vid:
                    audio_by_video.setdefault(vid, []).append(payload)

            return audio_by_video
        except Exception as e:
            logger.debug(f"Audio context fetch skipped: {e}")
            return {}

    async def _hydrate_results(
        self, entries: List[Dict], db: AsyncSession, decomp
    ) -> List[SearchResult]:
        """Enrich fused results with full video metadata from DB."""
        results = []
        video_cache = {}

        for entry in entries:
            vid = entry["video_id"]
            if vid not in video_cache:
                stmt = (
                    select(Video)
                    .where(Video.id == vid)
                    .options()
                )
                row = await db.execute(stmt)
                video = row.scalar_one_or_none()
                if video:
                    # Get channel name
                    ch_name = None
                    if video.channel_id:
                        ch_stmt = select(Channel.name).where(Channel.id == video.channel_id)
                        ch_row = await db.execute(ch_stmt)
                        ch_name = ch_row.scalar_one_or_none()
                    video_cache[vid] = (video, ch_name)

            if vid not in video_cache:
                continue

            video, channel_name = video_cache[vid]
            ts = entry["timestamp"]

            # Build explanation
            explanation = self._generate_explanation(entry, decomp)

            # Build timecoded URL (platform-aware)
            url = video.url
            tc_url = self._build_timecoded_url(url, ts)

            visual_tag_labels = []
            if entry.get("visual_tags"):
                for t in entry["visual_tags"][:5]:
                    if isinstance(t, dict):
                        visual_tag_labels.append(t.get("label", str(t)))
                    else:
                        visual_tag_labels.append(str(t))

            results.append(SearchResult(
                video_id=str(video.id),
                video_title=video.title,
                channel_name=channel_name,
                timestamp=ts,
                end_timestamp=entry.get("end_timestamp"),
                transcript_snippet=entry.get("transcript", "")[:300],
                visual_tags=visual_tag_labels,
                ocr_text=entry.get("ocr_text", "") or None,
                confidence_score=round(entry["final_score"], 4),
                modality_scores=ModalityScore(
                    text_semantic=round(entry.get("text_semantic", 0), 4),
                    visual_similarity=round(entry.get("visual_similarity", 0), 4),
                    ocr_match=round(entry.get("ocr_match", 0), 4),
                    keyword_match=round(entry.get("keyword_match", 0), 4),
                    temporal_coherence=round(entry.get("temporal_coherence", 0), 4),
                    emotion_context=round(entry.get("emotion_context", 0), 4),
                    comment_semantic=round(entry.get("comment_semantic", 0), 4),
                    comment_timestamp_boost=round(entry.get("comment_timestamp_boost", 0), 4),
                    agreement_cluster=round(entry.get("agreement_cluster", 0), 4),
                    entity_overlap=round(entry.get("entity_overlap", 0), 4),
                    user_cluster_confidence=round(entry.get("user_cluster_confidence", 0), 4),
                    audio_event_match=round(entry.get("audio_event_match", 0), 4),
                    audio_attribute_match=round(entry.get("audio_attribute_match", 0), 4),
                    alignment_quality=round(entry.get("alignment_quality", 0), 4),
                    change_point_proximity=round(entry.get("change_point_proximity", 0), 4),
                ),
                video_url=video.url,
                thumbnail_url=video.thumbnail_url,
                view_count=video.view_count,
                explanation=explanation,
                video_url_with_timecode=tc_url,
            ))

        return results

    @staticmethod
    def _build_timecoded_url(url: str, timestamp: float) -> str:
        """Build platform-aware timecoded URL."""
        t = int(timestamp)
        url_lower = url.lower()

        # YouTube: ?t=123s or &t=123s
        if "youtube.com" in url_lower or "youtu.be" in url_lower:
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}t={t}s"

        # Twitch: ?t=1h2m3s
        if "twitch.tv" in url_lower:
            h, m, s = t // 3600, (t % 3600) // 60, t % 60
            twitch_t = f"{h}h{m}m{s}s" if h > 0 else f"{m}m{s}s"
            sep = "&" if "?" in url else "?"
            return f"{url}{sep}t={twitch_t}"

        # Vimeo: #t=123s
        if "vimeo.com" in url_lower:
            return f"{url}#t={t}s"

        # TikTok, Instagram, Twitter — no URL timestamp support
        return url

    @staticmethod
    def _generate_explanation(entry: Dict, decomp) -> str:
        """Human-readable explanation of why this result matched."""
        reasons = []

        if entry.get("text_semantic", 0) > 0.3:
            reasons.append(f"Transcript semantically matches your query (score: {entry['text_semantic']:.2f})")
        if entry.get("keyword_match", 0) > 0:
            reasons.append("Keywords found in transcript")
        if entry.get("visual_similarity", 0) > 0.2:
            reasons.append(f"Visual content matches (score: {entry['visual_similarity']:.2f})")
        if entry.get("ocr_match", 0) > 0:
            reasons.append(f"On-screen text/numbers match: '{entry.get('ocr_text', '')[:50]}'")
        if entry.get("temporal_coherence", 0) > 0:
            reasons.append("Surrounding segments also match (temporal coherence)")

        if not reasons:
            reasons.append("Partial match across multiple signals")

        return " | ".join(reasons)

    async def _fetch_alignment_data(self, db: AsyncSession, video_ids: set = None) -> Dict[str, List[Dict]]:
        """Fetch transcript alignment data grouped by video_id."""
        if not video_ids:
            return {}
        try:
            result = await db.execute(
                select(TranscriptAlignment.video_id, TranscriptAlignment.start_time,
                       TranscriptAlignment.end_time, TranscriptAlignment.alignment_score,
                       TranscriptAlignment.quality_level)
                .where(TranscriptAlignment.video_id.in_(video_ids))
            )
            data: Dict[str, List[Dict]] = {}
            for row in result.all():
                vid = str(row[0])
                data.setdefault(vid, []).append({
                    "start_time": row[1], "end_time": row[2],
                    "alignment_score": row[3], "quality_level": row[4],
                })
            return data
        except Exception as e:
            logger.debug(f"Alignment data fetch skipped: {e}")
            return {}

    async def _fetch_change_point_data(self, db: AsyncSession, video_ids: set = None) -> Dict[str, List[float]]:
        """Fetch change point timestamps grouped by video_id."""
        if not video_ids:
            return {}
        try:
            result = await db.execute(
                select(AcousticChangePoint.video_id, AcousticChangePoint.timestamp)
                .where(AcousticChangePoint.magnitude >= 0.3)
                .where(AcousticChangePoint.video_id.in_(video_ids))
            )
            data: Dict[str, List[float]] = {}
            for row in result.all():
                vid = str(row[0])
                data.setdefault(vid, []).append(row[1])
            return data
        except Exception as e:
            logger.debug(f"Change point data fetch skipped: {e}")
            return {}

    def _build_vs_filters(self, filters: Optional[SearchFilters]) -> Optional[Dict]:
        if not filters:
            return None
        f = {}
        if filters.language:
            f["language"] = filters.language
        if filters.min_confidence:
            f["min_confidence"] = filters.min_confidence
        if filters.channels:
            f["channel_id"] = filters.channels[0]  # simplified
        return f if f else None


search_service = SearchService()
