"""
Nexum Intelligence Layer 4 — Interaction & Social Dynamics.

From comments and replies:
  Conversation: debate chains, echo chambers, sentiment cascades, authority patterns
  User profiles: positivity baseline, topic specialization, engagement velocity, loyalty
  Community: polarization index, meme velocity, narrative divergence, topic convergence
"""
from __future__ import annotations

import logging
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Conversation Structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DebateChain:
    """A detected argument/debate thread within comments."""
    root_comment_id: str
    video_id: str
    participant_count: int
    turn_count: int
    sentiment_polarity_range: float  # max - min sentiment across chain
    escalation_score: float          # 0-1: does sentiment get more extreme?
    resolution: str                  # "unresolved" | "consensus" | "abandoned" | "moderated"
    key_topics: List[str]
    confidence: float = 0.5


@dataclass
class EchoChamberCluster:
    """Group of users who consistently reinforce same viewpoints."""
    cluster_id: str
    video_ids: List[str]
    member_author_ids: List[str]
    avg_internal_sentiment_similarity: float  # 0-1
    dominant_sentiment: str           # "positive" | "negative" | "neutral"
    size: int
    insularity_score: float           # 0-1: how closed to outside views
    confidence: float = 0.5


@dataclass
class SentimentCascade:
    """Temporal propagation of sentiment through comment threads."""
    video_id: str
    trigger_comment_id: str
    cascade_type: str                 # "positive_wave" | "negative_spiral" | "controversy"
    affected_comments: int
    duration_minutes: float
    peak_intensity: float
    damping_rate: float               # how fast it decays
    confidence: float = 0.5


@dataclass
class AuthorityPattern:
    """Detected authority acknowledgment in comments."""
    video_id: str
    authority_author_id: str
    acknowledgment_count: int         # replies agreeing/deferring
    challenge_count: int              # replies disagreeing
    authority_score: float            # acknowledgment / (ack + challenge)
    topic_domain: Optional[str] = None
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# User Behavioral Profiles
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UserBehavioralProfile:
    """Comprehensive behavioral fingerprint for a comment author."""
    author_id: str
    positivity_baseline: float        # mean sentiment
    positivity_std: float
    topic_specialization_vector: List[float]  # topic affinity weights
    top_topics: List[str]
    engagement_velocity: float        # comments per day
    cross_video_loyalty: float        # 0-1: how much they stick to same channels
    avg_comment_length: float
    reply_ratio: float                # ratio of replies vs top-level
    time_of_day_pattern: List[float]  # 24-hour activity distribution
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# Community Metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CommunityMetrics:
    """Aggregate community health and dynamics metrics for a video or channel."""
    entity_id: str
    entity_type: str                  # "video" | "channel"
    polarization_index: float         # 0=consensus, 1=deeply split
    meme_velocity: float              # rate of repeated phrases/patterns
    narrative_divergence_score: float  # 0=single narrative, 1=competing narratives
    topic_convergence_speed: float    # how quickly comments converge on topic
    toxicity_ratio: float
    constructive_ratio: float
    unique_participant_count: int
    avg_thread_depth: float
    reply_reciprocity: float          # how often commenters reply to each other
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════


class SocialDynamicsService:
    """Analyzes interaction patterns and community dynamics from comment data."""

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("social_dynamics", True)
        logger.info("SocialDynamicsService initialized (enabled=%s)", self._enabled)

    def detect_debate_chains(
        self, video_id: str, comments: List[Dict]
    ) -> List[DebateChain]:
        """Identify debate/argument threads from comment trees."""
        try:
            threads = defaultdict(list)
            for c in comments:
                root = c.get("root_thread_id") or c.get("id")
                threads[root].append(c)

            debates = []
            for root_id, thread in threads.items():
                if len(thread) < 4:
                    continue

                sentiments = [(c.get("sentiment_score") or 0) for c in thread]
                participants = set(c.get("author_id") for c in thread if c.get("author_id"))

                if len(participants) < 2:
                    continue

                polarity_range = max(sentiments) - min(sentiments)
                if polarity_range < 0.3:
                    continue  # not a debate

                # Escalation: does sentiment variance increase over time?
                mid = len(sentiments) // 2
                first_half_std = float(np.std(sentiments[:mid])) if mid > 2 else 0.0
                second_half_std = float(np.std(sentiments[mid:])) if len(sentiments) - mid > 2 else 0.0
                escalation = min(1.0, max(0, (second_half_std - first_half_std) / max(first_half_std, 0.01)))

                # Resolution heuristic
                if len(thread) > 10 and abs(sentiments[-1]) < 0.1:
                    resolution = "consensus"
                elif len(thread) > 6 and thread[-1].get("depth_level", 0) == 0:
                    resolution = "abandoned"
                else:
                    resolution = "unresolved"

                topics = []
                for c in thread:
                    topics.extend(c.get("topic_labels") or [])
                top_topics = [t for t, _ in Counter(topics).most_common(3)]

                debates.append(DebateChain(
                    root_comment_id=root_id,
                    video_id=video_id,
                    participant_count=len(participants),
                    turn_count=len(thread),
                    sentiment_polarity_range=round(polarity_range, 3),
                    escalation_score=round(escalation, 3),
                    resolution=resolution,
                    key_topics=top_topics,
                    confidence=min(0.8, 0.3 + len(thread) * 0.02),
                ))

            return debates
        except Exception as _exc:
            logger.warning(f"SocialDynamicsService.detect_debate_chains failed: {_exc}")
            return None

    def compute_user_profile(
        self, author_id: str, comments: List[Dict], topic_names: List[str]
    ) -> UserBehavioralProfile:
        """Build behavioral profile from a user's comment history."""
        try:
            sentiments = [(c.get("sentiment_score") or 0) for c in comments]
            lengths = [len(c.get("text", "")) for c in comments]
            replies = sum(1 for c in comments if c.get("parent_comment_id"))

            # Topic vector
            topic_counts = Counter()
            for c in comments:
                for t in c.get("topic_labels") or []:
                    topic_counts[t] += 1
            total_topics = sum(topic_counts.values()) or 1
            topic_vec = [topic_counts.get(t, 0) / total_topics for t in topic_names[:32]]

            # Engagement velocity (comments per day)
            timestamps = sorted(c.get("timestamp_posted", "") for c in comments if c.get("timestamp_posted"))
            if len(timestamps) > 1:
                from datetime import datetime
                try:
                    first = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
                    last = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
                    days = max((last - first).total_seconds() / 86400, 1)
                    velocity = len(comments) / days
                except (ValueError, TypeError):
                    velocity = 0.5
            else:
                velocity = 0.0

            # Channel loyalty
            channels = set(c.get("channel_id") or c.get("video_id", "")[:8] for c in comments)
            loyalty = 1.0 / max(len(channels), 1)

            return UserBehavioralProfile(
                author_id=author_id,
                positivity_baseline=round(float(np.mean(sentiments)), 3) if sentiments else 0,
                positivity_std=round(float(np.std(sentiments)), 3) if sentiments else 0.3,
                topic_specialization_vector=topic_vec,
                top_topics=[t for t, _ in topic_counts.most_common(5)],
                engagement_velocity=round(velocity, 3),
                cross_video_loyalty=round(loyalty, 3),
                avg_comment_length=round(float(np.mean(lengths)), 1) if lengths else 0,
                reply_ratio=round(replies / max(len(comments), 1), 3),
                time_of_day_pattern=[1 / 24] * 24,  # would compute from timestamps
                confidence=min(0.85, 0.2 + len(comments) * 0.01),
            )
        except Exception as _exc:
            logger.warning(f"SocialDynamicsService.compute_user_profile failed: {_exc}")
            return None

    def compute_community_metrics(
        self, entity_id: str, entity_type: str, comments: List[Dict]
    ) -> CommunityMetrics:
        """Compute aggregate community health metrics."""
        try:
            if not comments:
                return CommunityMetrics(
                    entity_id=entity_id, entity_type=entity_type,
                    polarization_index=0, meme_velocity=0,
                    narrative_divergence_score=0, topic_convergence_speed=0,
                    toxicity_ratio=0, constructive_ratio=0.5,
                    unique_participant_count=0, avg_thread_depth=0,
                    reply_reciprocity=0, confidence=0.1,
                )

            sentiments = [(c.get("sentiment_score") or 0) for c in comments]
            participants = set(c.get("author_id") for c in comments if c.get("author_id"))
            depths = [c.get("depth_level", 0) for c in comments]
            toxicities = [c.get("toxicity_score", 0) for c in comments if c.get("toxicity_score") is not None]

            # Polarization: bimodal sentiment distribution
            if len(sentiments) > 10:
                pos_ratio = sum(1 for s in sentiments if s > 0.2) / len(sentiments)
                neg_ratio = sum(1 for s in sentiments if s < -0.2) / len(sentiments)
                polarization = min(1.0, 2 * min(pos_ratio, neg_ratio))
            else:
                polarization = 0.0

            # Meme velocity: repeated n-gram patterns
            texts = [c.get("text", "").lower() for c in comments]
            bigrams = Counter()
            for t in texts:
                words = t.split()
                for i in range(len(words) - 1):
                    bg = f"{words[i]} {words[i + 1]}"
                    bigrams[bg] += 1
            repeated = sum(1 for _, count in bigrams.items() if count > 3)
            meme_vel = min(1.0, repeated / max(len(comments) * 0.1, 1))

            # Narrative divergence from topic spread
            all_topics = []
            for c in comments:
                all_topics.extend(c.get("topic_labels") or [])
            topic_dist = Counter(all_topics)
            n_unique_topics = len(topic_dist)
            divergence = min(1.0, n_unique_topics / max(len(comments) * 0.05, 1))

            toxicity_ratio = float(np.mean(toxicities)) if toxicities else 0.0
            constructive = 1.0 - toxicity_ratio

            # Reply reciprocity
            reply_pairs = set()
            author_map = {c.get("id"): c.get("author_id") for c in comments}
            for c in comments:
                parent = c.get("parent_comment_id")
                if parent and parent in author_map:
                    pair = tuple(sorted([c.get("author_id", ""), author_map[parent]]))
                    reply_pairs.add(pair)
            reciprocity = min(1.0, len(reply_pairs) * 2 / max(len(participants), 1))

            return CommunityMetrics(
                entity_id=entity_id,
                entity_type=entity_type,
                polarization_index=round(polarization, 3),
                meme_velocity=round(meme_vel, 3),
                narrative_divergence_score=round(divergence, 3),
                topic_convergence_speed=round(1.0 - divergence, 3),
                toxicity_ratio=round(toxicity_ratio, 3),
                constructive_ratio=round(constructive, 3),
                unique_participant_count=len(participants),
                avg_thread_depth=round(float(np.mean(depths)), 2) if depths else 0,
                reply_reciprocity=round(reciprocity, 3),
                confidence=min(0.85, 0.2 + len(comments) * 0.002),
            )
        except Exception as _exc:
            logger.warning(f"SocialDynamicsService.compute_community_metrics failed: {_exc}")
            return None
