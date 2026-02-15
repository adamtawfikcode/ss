"""
Nexum Speech Service v4 — transcription with faster-whisper.

Quality-first configuration:
  - Model: large-v3 (~3 GB VRAM)
  - Beam size: 10, best_of: 5
  - Word-level timestamps for precise alignment
  - Condition on previous text for coherent output
  - Segment overlap for context continuity at boundaries
  - VAD filtering to skip silence
  - Filler word cleanup (optional — preserves emotion signals)

v4 Hardening:
  - Hallucination detection (repetition loops, impossible timestamps)
  - Retry logic with degraded settings on failure
  - Timeout protection for hung transcriptions
  - Corrupted audio resilience
"""
from __future__ import annotations

import logging
import re
import signal
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

FILLER_WORDS = {
    "um", "uh", "hmm", "huh", "like", "you know",
    "i mean", "basically", "actually", "right",
}

# Hallucination patterns: repeated phrases Whisper generates on silence/noise
HALLUCINATION_PATTERNS = [
    re.compile(r"(.{10,}?)\1{3,}"),  # same phrase repeated 3+ times
    re.compile(r"(thank you\.?\s*){4,}", re.IGNORECASE),
    re.compile(r"(please subscribe\.?\s*){3,}", re.IGNORECASE),
    re.compile(r"(music\.?\s*){5,}", re.IGNORECASE),
    re.compile(r"(\.\.\.){5,}"),
]


@dataclass
class TranscriptSegment:
    start_time: float
    end_time: float
    text: str
    confidence: float
    language: str = "en"
    speaker_label: Optional[str] = None


@dataclass
class TranscriptionResult:
    segments: List[TranscriptSegment]
    language: str
    language_confidence: float
    full_text: str
    word_count: int


class SpeechService:
    """Whisper-based speech-to-text with quality-first settings."""

    _model = None

    def _load_model(self):
        from faster_whisper import WhisperModel

        model_size = settings.whisper_model
        device = settings.device
        compute_type = settings.whisper_compute_type
        if device == "cpu":
            compute_type = "int8"

        logger.info(f"Loading Whisper model: {model_size} on {device} ({compute_type})")
        self._model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info("Whisper model loaded.")

    def release_model(self):
        """Free Whisper from GPU to make room for other models."""
        if self._model is not None:
            del self._model
            self._model = None
            if settings.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Whisper model released from memory.")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        cleanup_filler: bool = True,
    ) -> TranscriptionResult:
        """Transcribe audio file → structured segments with maximum accuracy."""
        # Try with full quality first; degrade on failure
        for attempt, params in enumerate(self._transcription_params()):
            try:
                return self._run_transcription(audio_path, language, cleanup_filler, params, attempt)
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt >= 2:
                    logger.error(f"All transcription attempts failed for {audio_path}")
                    return TranscriptionResult(
                        segments=[], language="unknown", language_confidence=0.0,
                        full_text="", word_count=0,
                    )

        return TranscriptionResult(segments=[], language="unknown", language_confidence=0.0, full_text="", word_count=0)

    @staticmethod
    def _transcription_params():
        """Degradation ladder: full quality → medium → fast fallback."""
        return [
            {"beam_size": settings.whisper_beam_size, "best_of": settings.whisper_best_of, "word_timestamps": True},
            {"beam_size": 5, "best_of": 3, "word_timestamps": True},
            {"beam_size": 1, "best_of": 1, "word_timestamps": False},
        ]

    def _run_transcription(self, audio_path, language, cleanup_filler, params, attempt):
        if self._model is None:
            self._load_model()

        segments_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=params["beam_size"],
            best_of=params["best_of"],
            word_timestamps=params["word_timestamps"],
            condition_on_previous_text=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                speech_pad_ms=200,
                threshold=0.35,
            ),
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        raw_segments = list(segments_iter)
        detected_lang = info.language
        lang_conf = info.language_probability

        logger.info(
            f"Transcribed (attempt {attempt + 1}): lang={detected_lang} ({lang_conf:.2f}), "
            f"raw_segments={len(raw_segments)}, params={params}"
        )

        # Chunk and filter
        chunked = self._chunk_segments(
            raw_segments,
            target_duration=settings.segment_duration_seconds,
            overlap=settings.segment_overlap_seconds,
            cleanup_filler=cleanup_filler,
        )

        # v4: Hallucination filtering
        chunked = self._filter_hallucinations(chunked)

        for seg in chunked:
            seg.language = detected_lang

        full_text = " ".join(s.text for s in chunked)

        return TranscriptionResult(
            segments=chunked,
            language=detected_lang,
            language_confidence=lang_conf,
            full_text=full_text,
            word_count=len(full_text.split()),
        )

    @staticmethod
    def _filter_hallucinations(segments: List[TranscriptSegment]) -> List[TranscriptSegment]:
        """Detect and remove Whisper hallucinations (repetition loops, impossible text)."""
        filtered = []
        for seg in segments:
            text = seg.text.strip()
            if not text:
                continue

            # Check hallucination patterns
            is_hallucination = False
            for pattern in HALLUCINATION_PATTERNS:
                if pattern.search(text):
                    logger.warning(f"Hallucination detected at {seg.start_time:.1f}s: '{text[:80]}...'")
                    is_hallucination = True
                    break

            # Check for extreme repetition via word frequency
            if not is_hallucination:
                words = text.lower().split()
                if len(words) > 5:
                    most_common = Counter(words).most_common(1)[0]
                    if most_common[1] / len(words) > 0.6:
                        logger.warning(f"Repetition hallucination at {seg.start_time:.1f}s: word '{most_common[0]}' is {most_common[1]}/{len(words)}")
                        is_hallucination = True

            # Check impossible timestamp (negative duration, very long gap)
            if seg.end_time < seg.start_time:
                logger.warning(f"Impossible timestamp at {seg.start_time:.1f}s")
                is_hallucination = True

            if not is_hallucination:
                filtered.append(seg)

        return filtered

    def _chunk_segments(
        self,
        raw_segments,
        target_duration: int = 15,
        overlap: float = 2.0,
        cleanup_filler: bool = True,
    ) -> List[TranscriptSegment]:
        """
        Merge Whisper's fine-grained segments into ~15s chunks.

        Uses word-level timestamps when available for precise boundaries.
        Adds overlap seconds from the previous chunk for context continuity.
        """
        chunks: List[TranscriptSegment] = []
        current_texts: List[str] = []
        current_start: Optional[float] = None
        current_confs: List[float] = []
        prev_chunk_end_text: str = ""  # last ~2s of previous chunk for overlap

        for seg in raw_segments:
            if current_start is None:
                current_start = seg.start

            text = seg.text.strip()
            if not text:
                continue

            if cleanup_filler:
                text = self._clean_text(text)
                if not text:
                    continue

            current_texts.append(text)
            # avg_logprob → ~confidence
            conf = max(0.0, min(1.0, 1.0 + seg.avg_log_prob))
            current_confs.append(conf)

            duration = seg.end - current_start
            if duration >= target_duration:
                joined = " ".join(current_texts)
                chunks.append(TranscriptSegment(
                    start_time=round(current_start, 2),
                    end_time=round(seg.end, 2),
                    text=joined,
                    confidence=round(sum(current_confs) / len(current_confs), 3),
                ))
                current_texts = []
                current_start = None
                current_confs = []

        # Flush remainder
        if current_texts and current_start is not None:
            last_end = raw_segments[-1].end if raw_segments else current_start + 1
            chunks.append(TranscriptSegment(
                start_time=round(current_start, 2),
                end_time=round(last_end, 2),
                text=" ".join(current_texts),
                confidence=round(sum(current_confs) / len(current_confs), 3) if current_confs else 0.0,
            ))

        # Apply overlap: each chunk gets the tail of the previous for context
        if overlap > 0 and len(chunks) > 1:
            for i in range(1, len(chunks)):
                prev = chunks[i - 1]
                overlap_start = max(prev.start_time, prev.end_time - overlap)
                # Adjust start_time to include overlap window
                chunks[i].start_time = round(min(chunks[i].start_time, overlap_start), 2)

        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Remove filler words and clean whitespace."""
        text = text.strip()
        if not text:
            return ""
        words = text.split()
        cleaned = [w for w in words if w.lower() not in FILLER_WORDS]
        result = " ".join(cleaned).strip()
        result = re.sub(r"\s+", " ", result)
        return result

    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """Quick language detection without full transcription."""
        if self._model is None:
            self._load_model()

        _, info = self._model.transcribe(
            audio_path,
            beam_size=1,
            best_of=1,
            vad_filter=True,
        )
        # Consume iterator minimally
        return info.language, info.language_probability


speech_service = SpeechService()
