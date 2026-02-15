"""
Nexum Intelligence Layers 5–8 — Combined Module.

Layer 5: Semantic Intent (intent detection, emotional arc, persuasion, certainty, bias)
Layer 6: Cross-Video Evolution (longitudinal creator tracking)
Layer 7: Graph-Derived Intelligence (centrality, bridges, propagation)
Layer 8: Meta-Quality Metrics (system-level reliability scores)
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# LAYER 5 — Semantic Intent
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SemanticIntentProfile:
    """Intent and rhetorical analysis of video content."""
    video_id: str
    # Intent detection (probabilities, sum to ~1)
    intent_scores: Dict[str, float]   # tutorial, satire, rant, news, ad, review, vlog, educational
    primary_intent: str
    # Emotional arc
    emotional_arc_type: str           # "ramp_up" | "cool_down" | "roller_coaster" | "flat" | "u_shape"
    emotional_arc_points: List[Tuple[float, float]]  # (time_ratio, valence)
    # Persuasion & rhetoric
    persuasion_style: str             # "logical" | "emotional" | "authority" | "social_proof" | "scarcity"
    persuasion_intensity: float       # 0-1
    certainty_ratio: float            # 0=all hedging, 1=all definitive claims
    bias_framing_polarity: float      # -1=negative framing, 0=neutral, 1=positive framing
    rhetorical_question_density: float  # per minute
    call_to_action_count: int
    confidence: float = 0.5


class SemanticIntentService:
    """Detects intent, emotional arcs, persuasion style from transcript + audio features."""

    INTENT_KEYWORDS = {
        "tutorial": {"how to", "step by step", "tutorial", "guide", "learn", "install", "setup", "walkthrough"},
        "satire": {"imagine", "obviously", "totally", "clearly joking", "satire", "parody"},
        "rant": {"annoyed", "frustrated", "sick of", "why is", "ridiculous", "unacceptable", "rant"},
        "news": {"breaking", "reports", "according to", "officials", "update", "coverage", "investigation"},
        "advertisement": {"sponsored", "discount", "code", "link below", "check out", "offer", "free trial"},
        "review": {"review", "rating", "pros and cons", "worth it", "recommend", "compared to", "unboxing"},
        "educational": {"explain", "theory", "concept", "principle", "understand", "definition", "fundamentally"},
        "vlog": {"today I", "my day", "went to", "hanging out", "morning routine", "what I ate"},
    }

    PERSUASION_KEYWORDS = {
        "logical": {"therefore", "because", "evidence", "data", "study", "research", "proves", "logically"},
        "emotional": {"feel", "imagine", "heartbreaking", "amazing", "incredible", "devastating", "beautiful"},
        "authority": {"expert", "professor", "according to", "PhD", "published", "credentials", "years of experience"},
        "social_proof": {"everyone", "millions", "trending", "popular", "most people", "community"},
        "scarcity": {"limited", "exclusive", "only", "running out", "hurry", "last chance", "rare"},
    }

    CERTAINTY_WORDS = {"definitely", "certainly", "absolutely", "clearly", "obviously", "undoubtedly", "always", "never", "fact"}
    HEDGING_WORDS = {"maybe", "perhaps", "might", "could", "possibly", "seems", "apparently", "arguably", "I think"}

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("semantic_intent", True)

    def analyze(self, video_id: str, transcript: str, segments: List[Dict] = None, duration_s: float = 300) -> SemanticIntentProfile:
        try:
            text_lower = transcript.lower()
            words = text_lower.split()
            n_words = max(len(words), 1)

            # Intent scoring
            intent_scores = {}
            for intent, kws in self.INTENT_KEYWORDS.items():
                hits = sum(1 for kw in kws if kw in text_lower)
                intent_scores[intent] = round(hits / max(len(kws), 1), 3)
            total = sum(intent_scores.values()) or 1
            intent_scores = {k: round(v / total, 3) for k, v in intent_scores.items()}
            primary = max(intent_scores, key=intent_scores.get)

            # Emotional arc from segment sentiments
            if segments:
                sentiments = [(s.get("start_time", 0) / max(duration_s, 1), s.get("sentiment", 0)) for s in segments if "sentiment" in s]
                if not sentiments:
                    sentiments = [(i / max(len(segments), 1), 0) for i in range(min(10, len(segments)))]
            else:
                sentiments = [(0, 0), (0.5, 0.1), (1.0, 0)]

            # Arc type classification
            if len(sentiments) > 3:
                vals = [s[1] for s in sentiments]
                first_q = np.mean(vals[:len(vals) // 3]) if vals else 0
                last_q = np.mean(vals[2 * len(vals) // 3:]) if vals else 0
                mid_q = np.mean(vals[len(vals) // 3:2 * len(vals) // 3]) if vals else 0

                if last_q > first_q + 0.15:
                    arc_type = "ramp_up"
                elif first_q > last_q + 0.15:
                    arc_type = "cool_down"
                elif mid_q < first_q - 0.1 and mid_q < last_q - 0.1:
                    arc_type = "u_shape"
                elif float(np.std(vals)) > 0.2:
                    arc_type = "roller_coaster"
                else:
                    arc_type = "flat"
            else:
                arc_type = "flat"

            # Persuasion style
            persuasion_scores = {}
            for style, kws in self.PERSUASION_KEYWORDS.items():
                persuasion_scores[style] = sum(1 for kw in kws if kw in text_lower) / max(len(kws), 1)
            persuasion_style = max(persuasion_scores, key=persuasion_scores.get)
            persuasion_intensity = min(1.0, max(persuasion_scores.values()) * 2)

            # Certainty vs hedging
            cert_count = sum(1 for w in words if w in self.CERTAINTY_WORDS)
            hedge_count = sum(1 for w in words if w in self.HEDGING_WORDS)
            certainty_ratio = cert_count / max(cert_count + hedge_count, 1)

            # Bias framing
            positive_frames = sum(1 for w in words if w in {"great", "excellent", "wonderful", "best", "amazing", "incredible"})
            negative_frames = sum(1 for w in words if w in {"terrible", "worst", "horrible", "disgusting", "awful", "pathetic"})
            bias = (positive_frames - negative_frames) / max(positive_frames + negative_frames, 1)

            # Rhetorical questions
            rq_count = text_lower.count("?")
            rq_density = rq_count / max(duration_s / 60, 1)

            # CTAs
            cta_phrases = ["subscribe", "hit the bell", "like this video", "comment below", "share", "link in", "check out"]
            cta_count = sum(1 for p in cta_phrases if p in text_lower)

            return SemanticIntentProfile(
                video_id=video_id, intent_scores=intent_scores, primary_intent=primary,
                emotional_arc_type=arc_type, emotional_arc_points=sentiments[:20],
                persuasion_style=persuasion_style, persuasion_intensity=round(persuasion_intensity, 3),
                certainty_ratio=round(certainty_ratio, 3),
                bias_framing_polarity=round(bias, 3),
                rhetorical_question_density=round(rq_density, 2),
                call_to_action_count=cta_count,
                confidence=min(0.85, 0.3 + n_words * 0.0001),
            )
        except Exception as _exc:
            logger.warning(f"SemanticIntentService.analyze failed: {_exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# LAYER 6 — Cross-Video Evolution
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CreatorEvolutionProfile:
    """Longitudinal tracking of creator changes over time."""
    channel_id: str
    time_range: Tuple[str, str]       # (first_video_date, latest_video_date)
    video_count: int
    # Drift metrics (positive = increasing)
    intro_length_drift: float         # seconds/video trend
    speech_speed_drift: float         # wpm trend per video
    vocabulary_growth_rate: float     # new unique words per video
    thumbnail_style_drift: float      # palette distance evolution
    sponsor_density_trend: float      # sponsors per video trend
    editing_complexity_trend: float   # cuts per minute trend
    camera_quality_drift: float       # resolution/clarity trend
    production_value_curve: List[float]  # per-video score over time
    # Phase detection
    phases: List[Dict]                # [{"start_idx": 0, "end_idx": 15, "label": "amateur"}]
    confidence: float = 0.5


class CrossVideoEvolutionService:
    """Tracks longitudinal creator evolution across video chronology."""

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("cross_video_evolution", True)

    def compute_evolution(
        self, channel_id: str, videos_chronological: List[Dict]
    ) -> CreatorEvolutionProfile:
        """Compute evolution metrics from chronologically sorted video data."""
        try:
            n = len(videos_chronological)
            if n < 3:
                return CreatorEvolutionProfile(
                    channel_id=channel_id, time_range=("", ""), video_count=n,
                    intro_length_drift=0, speech_speed_drift=0, vocabulary_growth_rate=0,
                    thumbnail_style_drift=0, sponsor_density_trend=0,
                    editing_complexity_trend=0, camera_quality_drift=0,
                    production_value_curve=[], phases=[], confidence=0.1,
                )

            # Extract time series from video metadata
            intro_lengths = []
            speech_speeds = []
            vocab_sizes = []
            sponsor_counts = []
            cut_rates = []
            prod_scores = []

            for v in videos_chronological:
                meta = v.get("metadata_json") or {}
                signals = v.get("signals") or {}

                intro_lengths.append(signals.get("intro_duration_s", 5.0))
                speech_speeds.append(signals.get("speaking_rate_wpm", 150))
                vocab_sizes.append(signals.get("unique_word_count", 500))
                sponsor_counts.append(1 if signals.get("has_sponsor") else 0)
                cut_rates.append(signals.get("cut_frequency_per_min", 5))

                # Composite production value
                score = (
                    min(1, signals.get("mic_quality_score", 0.5)) * 0.25 +
                    min(1, cut_rates[-1] / 20) * 0.25 +
                    (1.0 if v.get("has_captions") else 0.5) * 0.25 +
                    min(1, vocab_sizes[-1] / 2000) * 0.25
                )
                prod_scores.append(round(score, 3))

            # Linear regression slopes
            x = np.arange(n)
            intro_drift = float(np.polyfit(x, intro_lengths, 1)[0]) if n > 2 else 0
            speed_drift = float(np.polyfit(x, speech_speeds, 1)[0]) if n > 2 else 0
            vocab_growth = float(np.polyfit(x, vocab_sizes, 1)[0]) if n > 2 else 0
            sponsor_trend = float(np.polyfit(x, sponsor_counts, 1)[0]) if n > 2 else 0
            edit_trend = float(np.polyfit(x, cut_rates, 1)[0]) if n > 2 else 0

            # Phase detection (simple: split into thirds)
            phases = []
            third = max(n // 3, 1)
            for i, (start, end, label) in enumerate([
                (0, third, "early"),
                (third, 2 * third, "mid"),
                (2 * third, n, "recent"),
            ]):
                slice_data = prod_scores[start:end]
                avg_prod = float(np.mean(slice_data)) if len(slice_data) > 0 else 0.0
                phases.append({"start_idx": start, "end_idx": end, "label": label,
                              "avg_production_score": round(avg_prod, 3)})

            dates = [v.get("uploaded_at") or v.get("created_at", "") for v in videos_chronological]

            return CreatorEvolutionProfile(
                channel_id=channel_id,
                time_range=(dates[0] if dates else "", dates[-1] if dates else ""),
                video_count=n,
                intro_length_drift=round(intro_drift, 4),
                speech_speed_drift=round(speed_drift, 4),
                vocabulary_growth_rate=round(vocab_growth, 2),
                thumbnail_style_drift=0.0,  # would need thumbnail embeddings
                sponsor_density_trend=round(sponsor_trend, 4),
                editing_complexity_trend=round(edit_trend, 4),
                camera_quality_drift=0.0,
                production_value_curve=prod_scores,
                phases=phases,
                confidence=min(0.85, 0.3 + n * 0.01),
            )
        except Exception as _exc:
            logger.warning(f"CrossVideoEvolutionService.compute_evolution failed: {_exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# LAYER 7 — Graph-Derived Intelligence
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GraphIntelligence:
    """Intelligence derived from knowledge graph topology."""
    entity_id: str
    entity_type: str
    influence_centrality: float       # PageRank-style
    betweenness_centrality: float     # bridge importance
    is_bridge_node: bool              # connects otherwise disconnected communities
    community_id: Optional[str] = None
    # Propagation
    narrative_propagation_velocity: float = 0.0  # avg hops/day
    # Affinity vectors
    topic_affinity_vector: List[float] = field(default_factory=list)
    reciprocity_imbalance: float = 0.0  # how asymmetric connections are
    confidence: float = 0.5


class GraphIntelligenceService:
    """Computes intelligence metrics from the knowledge graph structure."""

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("graph_intelligence", True)

    def compute_centralities(
        self, nodes: List[Dict], edges: List[Dict]
    ) -> Dict[str, GraphIntelligence]:
        """Compute centrality metrics for all nodes."""
        if not nodes:
            return {}
        # Build adjacency
        try:
            adj: Dict[str, List[str]] = defaultdict(list)
            in_degree: Dict[str, int] = defaultdict(int)
            out_degree: Dict[str, int] = defaultdict(int)
            node_types = {n["id"]: n.get("node_type", "Unknown") for n in nodes}

            for e in edges:
                s, t = e.get("source"), e.get("target")
                if s and t:
                    adj[s].append(t)
                    adj[t].append(s)
                    out_degree[s] += 1
                    in_degree[t] += 1

            all_ids = set(n["id"] for n in nodes)
            n_nodes = max(len(all_ids), 1)

            # Simplified PageRank (power iteration, 10 iterations)
            pr = {nid: 1.0 / n_nodes for nid in all_ids}
            damping = 0.85
            for _ in range(10):
                new_pr = {}
                for nid in all_ids:
                    incoming_sum = sum(
                        pr.get(src, 0) / max(out_degree.get(src, 1), 1)
                        for src in adj[nid]
                    )
                    new_pr[nid] = (1 - damping) / n_nodes + damping * incoming_sum
                pr = new_pr

            # Normalize to 0-1
            max_pr = max(pr.values()) or 1
            pr = {k: v / max_pr for k, v in pr.items()}

            # Betweenness approximation: nodes with high degree connecting different communities
            results = {}
            for nid in all_ids:
                degree = len(adj[nid])
                neighbors = set(adj[nid])
                # Bridge detection: low clustering coefficient + high degree
                if degree > 0:
                    neighbor_edges = sum(1 for n1 in neighbors for n2 in adj[n1] if n2 in neighbors)
                    max_possible = degree * (degree - 1)
                    clustering = neighbor_edges / max(max_possible, 1)
                    betweenness = (1 - clustering) * degree / max(n_nodes, 1)
                    is_bridge = clustering < 0.2 and degree > 3
                else:
                    betweenness = 0
                    is_bridge = False

                # Reciprocity imbalance
                in_d = in_degree.get(nid, 0)
                out_d = out_degree.get(nid, 0)
                total_d = in_d + out_d
                imbalance = abs(in_d - out_d) / max(total_d, 1)

                results[nid] = GraphIntelligence(
                    entity_id=nid,
                    entity_type=node_types.get(nid, "Unknown"),
                    influence_centrality=round(pr.get(nid, 0), 4),
                    betweenness_centrality=round(min(1, betweenness), 4),
                    is_bridge_node=is_bridge,
                    reciprocity_imbalance=round(imbalance, 3),
                    confidence=min(0.85, 0.3 + n_nodes * 0.0001),
                )

            return results
        except Exception as _exc:
            logger.warning(f"GraphIntelligenceService.compute_centralities failed: {_exc}")
            return None


# ═══════════════════════════════════════════════════════════════════════
# LAYER 8 — Meta-Quality Metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MetaQualityReport:
    """System-level quality assessment of processed content."""
    video_id: str
    transcript_reliability_index: float  # 0-1: how trustworthy is our transcript?
    visual_clarity_index: float          # 0-1: quality of visual extraction
    audio_cleanliness_score: float       # 0-1: audio processing quality
    editing_professionalism_score: float # 0-1: production value assessment
    redundancy_index: float              # 0=unique, 1=highly repetitive
    information_density_score: float     # 0=filler, 1=dense information
    overall_quality: float               # weighted composite
    data_completeness: float             # 0-1: how much data we have
    confidence: float = 0.5


class MetaQualityService:
    """Evaluates the quality and completeness of our processed data."""

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("meta_quality", True)

    def evaluate(
        self, video_id: str,
        segments: List[Dict] = None,
        frames: List[Dict] = None,
        audio_segments: List[Dict] = None,
        comments: List[Dict] = None,
        duration_s: float = 300,
    ) -> MetaQualityReport:
        try:
            segments = segments or []
            frames = frames or []
            audio_segments = audio_segments or []

            # Transcript reliability: confidence scores + coverage
            if segments:
                confs = [s.get("confidence", 0.5) for s in segments]
                coverage = sum(s.get("end_time", 0) - s.get("start_time", 0) for s in segments) / max(duration_s, 1)
                transcript_rel = float(np.mean(confs)) * 0.6 + min(coverage, 1) * 0.4
            else:
                transcript_rel = 0.0

            # Visual clarity: OCR confidence + frame coverage
            if frames:
                ocr_confs = [f.get("ocr_confidence", 0.5) for f in frames if f.get("ocr_confidence")]
                vis_clarity = float(np.mean(ocr_confs)) if ocr_confs else 0.3
                vis_clarity = vis_clarity * 0.5 + min(len(frames) / max(duration_s / 5, 1), 1) * 0.5
            else:
                vis_clarity = 0.0

            # Audio cleanliness
            if audio_segments:
                loudness_vals = [s.get("loudness_lufs", -30) for s in audio_segments if s.get("loudness_lufs")]
                noise_floors = [s.get("noise_floor_db", -40) for s in audio_segments if "noise_floor_db" in s]
                cleanliness = 0.5
                if loudness_vals:
                    consistency = 1.0 - min(float(np.std(loudness_vals)) / 10, 1)
                    cleanliness = consistency * 0.5 + 0.5
                if noise_floors:
                    cleanliness = cleanliness * 0.5 + min(1, (float(np.mean(noise_floors)) + 60) / 40) * 0.5
            else:
                cleanliness = 0.0

            # Editing professionalism: scene changes, audio consistency, caption availability
            editing = 0.5
            scene_changes = sum(1 for f in frames if f.get("is_scene_change"))
            if duration_s > 0 and scene_changes > 0:
                cut_rate = scene_changes / (duration_s / 60)
                editing = min(1.0, cut_rate / 15) * 0.3 + cleanliness * 0.4 + vis_clarity * 0.3

            # Redundancy: repeated n-grams in transcript
            if segments:
                all_text = " ".join(s.get("text", "") for s in segments).lower().split()
                if len(all_text) > 50:
                    trigrams = [tuple(all_text[i:i + 3]) for i in range(len(all_text) - 2)]
                    tc = Counter(trigrams)
                    repeat_ratio = sum(1 for _, c in tc.items() if c > 2) / max(len(tc), 1)
                    redundancy = min(1.0, repeat_ratio * 5)
                else:
                    redundancy = 0.3
            else:
                redundancy = 0.5

            # Information density
            if segments and duration_s > 0:
                total_words = sum(len(s.get("text", "").split()) for s in segments)
                wpm = total_words / max(duration_s / 60, 1)
                info_density = min(1.0, wpm / 180) * (1 - redundancy * 0.5)
            else:
                info_density = 0.0

            # Data completeness
            has = [bool(segments), bool(frames), bool(audio_segments), bool(comments)]
            completeness = sum(has) / len(has)

            overall = (
                transcript_rel * 0.2 + vis_clarity * 0.15 + cleanliness * 0.15 +
                editing * 0.15 + (1 - redundancy) * 0.1 + info_density * 0.15 +
                completeness * 0.1
            )

            return MetaQualityReport(
                video_id=video_id,
                transcript_reliability_index=round(transcript_rel, 3),
                visual_clarity_index=round(vis_clarity, 3),
                audio_cleanliness_score=round(cleanliness, 3),
                editing_professionalism_score=round(editing, 3),
                redundancy_index=round(redundancy, 3),
                information_density_score=round(info_density, 3),
                overall_quality=round(overall, 3),
                data_completeness=round(completeness, 3),
                confidence=min(0.85, completeness * 0.7 + 0.15),
            )
        except Exception as _exc:
            logger.warning(f"MetaQualityService.evaluate failed: {_exc}")
            return None
