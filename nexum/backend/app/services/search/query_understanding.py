"""
Nexum Query Understanding Pipeline

Decomposes natural-language queries into structured search dimensions:
- Objects, Actions, Numbers, Time/era, Emotion, Context

Each dimension is searched independently, then fused during ranking.
"""
from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# ── Lexicons ─────────────────────────────────────────────────────────────

EMOTION_WORDS = {
    "happy", "sad", "angry", "frustrated", "excited", "scared", "surprised",
    "confused", "bored", "struggling", "laughing", "crying", "arguing",
    "shouting", "whispering", "nervous", "calm", "panicking", "celebrating",
    "depressed", "anxious", "joyful", "furious", "terrified",
}

ACTION_WORDS = {
    "playing", "cooking", "eating", "running", "walking", "talking",
    "explaining", "presenting", "coding", "typing", "drawing", "painting",
    "singing", "dancing", "fighting", "arguing", "debating", "reviewing",
    "unboxing", "building", "fixing", "breaking", "stealing", "throwing",
    "catching", "jumping", "swimming", "driving", "flying", "reading",
    "writing", "watching", "streaming", "recording", "interviewing",
    "teaching", "learning", "studying", "working", "sleeping",
    "discussing", "showing", "demonstrating",
}

TIME_ERA_PATTERNS = [
    (r"\b(retro|vintage|old school|classic|90s|80s|70s|60s)\b", "retro"),
    (r"\b(modern|recent|new|2024|2023|2025|latest|current)\b", "modern"),
    (r"\b(night|nighttime|dark|evening)\b", "night"),
    (r"\b(daytime|morning|afternoon|bright)\b", "daytime"),
]

CONTEXT_PATTERNS = [
    (r"\b(podcast|interview|talk show|radio)\b", "podcast/interview"),
    (r"\b(tutorial|how.?to|guide|lesson|course)\b", "tutorial"),
    (r"\b(review|unboxing|first look|hands.?on)\b", "review"),
    (r"\b(vlog|daily|routine|day in)\b", "vlog"),
    (r"\b(gaming|gameplay|playthrough|lets? play)\b", "gaming"),
    (r"\b(news|breaking|report|coverage)\b", "news"),
    (r"\b(music|song|concert|performance|live)\b", "music"),
    (r"\b(sport|game|match|tournament|championship)\b", "sports"),
]


@dataclass
class QueryDecomposition:
    original: str
    objects: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    numbers: List[str] = field(default_factory=list)
    time_era: Optional[str] = None
    emotion: Optional[str] = None
    context: Optional[str] = None
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "objects": self.objects,
            "actions": self.actions,
            "numbers": self.numbers,
            "time_era": self.time_era,
            "emotion": self.emotion,
            "context": self.context,
        }


class QueryUnderstandingPipeline:
    """Decompose a user query into structured search dimensions."""

    def decompose(self, query: str) -> QueryDecomposition:
        query_lower = query.lower().strip()
        words = set(re.findall(r"\b\w+\b", query_lower))

        result = QueryDecomposition(original=query)

        # Extract numbers
        result.numbers = re.findall(r"\b\d[\d,.:]*\d|\b\d+\b", query_lower)

        # Extract emotions
        emotions = words & EMOTION_WORDS
        if emotions:
            result.emotion = sorted(emotions)[0]  # primary emotion

        # Extract actions
        result.actions = sorted(words & ACTION_WORDS)

        # Extract time/era
        for pattern, era in TIME_ERA_PATTERNS:
            if re.search(pattern, query_lower):
                result.time_era = era
                break

        # Extract context
        for pattern, ctx in CONTEXT_PATTERNS:
            if re.search(pattern, query_lower):
                result.context = ctx
                break

        # Extract objects (nouns that aren't actions/emotions/numbers/stopwords)
        stopwords = {
            "a", "an", "the", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "with", "by", "from", "up", "about", "into",
            "through", "during", "before", "after", "above", "below", "between",
            "where", "when", "who", "what", "how", "why", "which", "that",
            "this", "they", "them", "their", "he", "she", "it", "his", "her",
            "and", "but", "or", "not", "no", "so", "if", "then", "than",
            "very", "just", "really", "some", "any", "all", "each", "every",
            "guy", "person", "someone", "thing", "stuff",
        }
        number_words = set(re.findall(r"\b\d+\b", query_lower))
        candidate_objects = words - ACTION_WORDS - EMOTION_WORDS - stopwords - number_words
        result.objects = sorted(candidate_objects)

        # Build keyword list (all meaningful tokens)
        result.keywords = sorted(
            set(result.objects + result.actions + result.numbers)
        )

        logger.debug(f"Query decomposed: {result}")
        return result

    def build_search_queries(self, decomp: QueryDecomposition) -> dict:
        """Build specialized sub-queries for each modality."""
        queries = {
            "semantic": decomp.original,
            "keyword": " ".join(decomp.keywords),
        }

        if decomp.numbers:
            queries["ocr"] = " ".join(decomp.numbers)

        if decomp.objects:
            queries["visual"] = " ".join(decomp.objects)

        if decomp.emotion:
            queries["emotion"] = decomp.emotion

        return queries


query_pipeline = QueryUnderstandingPipeline()
