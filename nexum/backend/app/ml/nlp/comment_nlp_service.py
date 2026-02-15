"""
Nexum Comment NLP Service

Processes each comment through:
  1. Embedding generation (all-mpnet-base-v2, 768-dim)
  2. Sentiment analysis (positive/negative/neutral + score)
  3. Named Entity Recognition (spaCy)
  4. Topic classification (zero-shot via sentence-transformers)
  5. Timestamp extraction ("1:23", "at 5 minutes")
  6. Numeric token extraction
  7. Toxicity / spam scoring (heuristic + model)
  8. Language detection

All GPU models use the same sequential loading strategy as the media pipeline.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ── Timestamp patterns ───────────────────────────────────────────────────

TIMESTAMP_PATTERNS = [
    re.compile(r"\b(\d{1,2}):(\d{2}):(\d{2})\b"),           # 1:23:45
    re.compile(r"\b(\d{1,2}):(\d{2})\b"),                    # 1:23
    re.compile(r"\bat\s+(\d+)\s*(?:min(?:ute)?s?)\b", re.I), # at 5 minutes
    re.compile(r"\b(\d+)\s*(?:sec(?:ond)?s?)\b", re.I),      # 30 seconds
]

NUMBER_PATTERN = re.compile(r"\b\d[\d,.:]*\d|\b\d+\b")

# ── Spam heuristics ──────────────────────────────────────────────────────

SPAM_INDICATORS = [
    re.compile(r"(subscribe|sub to|check out my|follow me|promo code)", re.I),
    re.compile(r"(free\s+(?:v-?bucks|gift|iphone|money))", re.I),
    re.compile(r"(https?://\S+){2,}", re.I),  # multiple URLs
    re.compile(r"(.)\1{7,}"),  # character repetition (aaaaaaaaaa)
    re.compile(r"[A-Z\s]{30,}"),  # excessive caps
]


@dataclass
class CommentNLPResult:
    """Output of NLP processing for a single comment."""
    embedding: Optional[np.ndarray] = None
    sentiment_score: float = 0.0          # -1.0 to 1.0
    sentiment_label: str = "neutral"       # positive | negative | neutral
    entities: List[Dict] = field(default_factory=list)  # [{name, type, start, end}]
    topic_labels: List[str] = field(default_factory=list)
    topic_scores: List[float] = field(default_factory=list)
    extracted_timestamps: List[str] = field(default_factory=list)   # raw strings
    extracted_timestamp_seconds: List[float] = field(default_factory=list)
    extracted_numbers: List[str] = field(default_factory=list)
    toxicity_score: float = 0.0           # 0.0 to 1.0
    spam_score: float = 0.0              # 0.0 to 1.0
    language: Optional[str] = None


class CommentNLPService:
    """Lazy-loaded NLP pipeline for comment analysis."""

    def __init__(self):
        self._embedding_model = None
        self._sentiment_pipeline = None
        self._ner_model = None
        self._topic_model = None
        self._initialized = False

    def _ensure_loaded(self):
        """Lazy-load all NLP models on first use."""
        if self._initialized:
            return

        logger.info("Loading Comment NLP models...")

        # 1. Embedding model (reuse from embedding_service if available)
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                settings.text_embedding_model,
                device=settings.device,
            )
            logger.info(f"Loaded embedding model: {settings.text_embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")

        # 2. Sentiment analysis
        try:
            from transformers import pipeline as hf_pipeline
            self._sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=settings.sentiment_model,
                device=0 if settings.device == "cuda" else -1,
                truncation=True,
                max_length=512,
            )
            logger.info(f"Loaded sentiment model: {settings.sentiment_model}")
        except Exception as e:
            logger.warning(f"Sentiment model unavailable: {e}")

        # 3. spaCy NER
        try:
            import spacy
            self._ner_model = spacy.load(settings.ner_model)
            logger.info(f"Loaded spaCy NER: {settings.ner_model}")
        except Exception as e:
            logger.warning(f"spaCy NER unavailable: {e}")

        # 4. Topic classification (zero-shot)
        try:
            from transformers import pipeline as hf_pipeline
            self._topic_model = hf_pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if settings.device == "cuda" else -1,
            )
            logger.info("Loaded zero-shot topic classifier")
        except Exception as e:
            logger.warning(f"Topic classifier unavailable: {e}")

        self._initialized = True
        logger.info("Comment NLP models loaded")

    # ── Main Processing ──────────────────────────────────────────────────

    def process_comment(self, text: str) -> CommentNLPResult:
        """Run full NLP pipeline on a single comment."""
        self._ensure_loaded()

        result = CommentNLPResult()

        if not text or not text.strip():
            return result

        clean_text = text.strip()[:2000]  # Cap input length

        # 1. Embedding
        result.embedding = self._generate_embedding(clean_text)

        # 2. Sentiment
        sent_score, sent_label = self._analyze_sentiment(clean_text)
        result.sentiment_score = sent_score
        result.sentiment_label = sent_label

        # 3. NER
        result.entities = self._extract_entities(clean_text)

        # 4. Topic classification
        result.topic_labels, result.topic_scores = self._classify_topics(clean_text)

        # 5. Timestamps
        result.extracted_timestamps, result.extracted_timestamp_seconds = (
            self._extract_timestamps(clean_text)
        )

        # 6. Numbers
        result.extracted_numbers = self._extract_numbers(clean_text)

        # 7. Toxicity / spam
        result.toxicity_score = self._score_toxicity(clean_text)
        result.spam_score = self._score_spam(clean_text)

        # 8. Language detection
        result.language = self._detect_language(clean_text)

        return result

    def process_batch(self, texts: List[str]) -> List[CommentNLPResult]:
        """Process multiple comments — batch embedding, individual NLP."""
        self._ensure_loaded()
        results = []

        # Batch embeddings for efficiency
        embeddings = self._generate_embeddings_batch(texts)

        for i, text in enumerate(texts):
            result = self.process_comment(text)
            if embeddings is not None and i < len(embeddings):
                result.embedding = embeddings[i]
            results.append(result)

        return results

    # ── Sub-Processors ───────────────────────────────────────────────────

    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        if not self._embedding_model:
            return None
        try:
            return self._embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return None

    def _generate_embeddings_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self._embedding_model or not texts:
            return None
        try:
            clean = [t.strip()[:2000] for t in texts if t and t.strip()]
            if not clean:
                return None
            return self._embedding_model.encode(
                clean, convert_to_numpy=True, normalize_embeddings=True,
                batch_size=32, show_progress_bar=False,
            )
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            return None

    def _analyze_sentiment(self, text: str) -> Tuple[float, str]:
        if not self._sentiment_pipeline:
            return 0.0, "neutral"
        try:
            result = self._sentiment_pipeline(text[:512])[0]
            label = result["label"].lower()
            score = result["score"]
            # Normalize to -1..1
            if "positive" in label:
                return score, "positive"
            elif "negative" in label:
                return -score, "negative"
            return 0.0, "neutral"
        except Exception:
            return 0.0, "neutral"

    def _extract_entities(self, text: str) -> List[Dict]:
        if not self._ner_model:
            return []
        try:
            doc = self._ner_model(text[:5000])
            entities = []
            seen = set()
            for ent in doc.ents:
                key = (ent.text.lower(), ent.label_)
                if key not in seen:
                    seen.add(key)
                    entities.append({
                        "name": ent.text,
                        "type": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    })
            return entities
        except Exception:
            return []

    def _classify_topics(self, text: str) -> Tuple[List[str], List[float]]:
        if not self._topic_model:
            return [], []
        try:
            result = self._topic_model(
                text[:512],
                candidate_labels=settings.topic_labels,
                multi_label=True,
            )
            labels = []
            scores = []
            for label, score in zip(result["labels"], result["scores"]):
                if score >= 0.3:
                    labels.append(label)
                    scores.append(round(score, 3))
            return labels[:5], scores[:5]
        except Exception:
            return [], []

    @staticmethod
    def _extract_timestamps(text: str) -> Tuple[List[str], List[float]]:
        raw_timestamps = []
        seconds_list = []

        for pattern in TIMESTAMP_PATTERNS:
            for match in pattern.finditer(text):
                raw_timestamps.append(match.group(0))
                groups = match.groups()
                try:
                    if len(groups) == 3:  # H:M:S
                        secs = int(groups[0]) * 3600 + int(groups[1]) * 60 + int(groups[2])
                    elif len(groups) == 2:  # M:S
                        secs = int(groups[0]) * 60 + int(groups[1])
                    else:
                        secs = int(groups[0])
                    seconds_list.append(float(secs))
                except (ValueError, IndexError):
                    pass

        return raw_timestamps, seconds_list

    @staticmethod
    def _extract_numbers(text: str) -> List[str]:
        return NUMBER_PATTERN.findall(text)

    @staticmethod
    def _score_toxicity(text: str) -> float:
        """Heuristic toxicity scoring (GPU model can be swapped in)."""
        score = 0.0
        text_lower = text.lower()

        # Excessive caps ratio
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.6:
                score += 0.3

        # Excessive punctuation
        excl_count = text.count("!") + text.count("?")
        if excl_count > 5:
            score += 0.2

        # Length-based (very short aggressive comments)
        if len(text.split()) <= 3 and any(c in text for c in "!?"):
            score += 0.1

        return min(score, 1.0)

    @staticmethod
    def _score_spam(text: str) -> float:
        """Heuristic spam scoring."""
        score = 0.0
        for pattern in SPAM_INDICATORS:
            if pattern.search(text):
                score += 0.25
        # Duplicate detection helper: very short generic comments
        word_count = len(text.split())
        if word_count <= 2:
            score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _detect_language(text: str) -> Optional[str]:
        """Simple language detection heuristic."""
        try:
            from langdetect import detect
            return detect(text[:500])
        except Exception:
            return None

    def release(self):
        """Release GPU memory."""
        self._embedding_model = None
        self._sentiment_pipeline = None
        self._topic_model = None
        self._initialized = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Comment NLP models released")


# Module-level singleton
comment_nlp_service = CommentNLPService()
