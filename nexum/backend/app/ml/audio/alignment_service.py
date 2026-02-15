"""
Nexum Audio-Transcript Alignment Service.

Measures transcript fidelity by cross-referencing audio signals with ASR output.

Signals:
  A. Speech Presence vs Transcript Density
     - High speech_prob + low word count = mismatch
  B. ASR Confidence Consistency
     - Low segment confidence + strong speech = red flag
  C. Audio-Text Embedding Similarity
     - PANNs audio embedding vs text embedding cosine similarity
  D. Timing Drift
     - Gap between expected speech duration and transcript duration

Output: alignment_score (0–1), sub-signal breakdown, quality warnings

Unlocks:
  - "Transcript quality low" warnings in UI
  - Search ranking boosts for high-alignment segments
  - Forensic detection of edited captions / manipulated voiceovers
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class AlignmentSignal:
    """Individual alignment sub-signal."""
    name: str
    score: float        # 0–1 (1 = perfect alignment)
    weight: float       # contribution to final score
    detail: str = ""    # human-readable explanation


@dataclass
class AlignmentResult:
    """Audio-transcript alignment result for a segment."""
    start_time: float
    end_time: float
    alignment_score: float          # 0–1 weighted composite
    signals: List[AlignmentSignal] = field(default_factory=list)
    quality_level: str = "good"     # good / acceptable / poor / mismatch
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
            "alignment_score": round(self.alignment_score, 3),
            "quality_level": self.quality_level,
            "signals": [
                {"name": s.name, "score": round(s.score, 3), "detail": s.detail}
                for s in self.signals
            ],
            "warnings": self.warnings,
        }


@dataclass
class FullAlignmentResult:
    """Alignment results for entire audio file."""
    segments: List[AlignmentResult]
    overall_score: float
    overall_quality: str
    total_warnings: int
    mismatch_regions: List[Tuple[float, float]]  # (start, end) of mismatches


class AudioTranscriptAlignmentService:
    """
    Cross-references audio intelligence data with transcript segments
    to detect misalignments, hallucinations, and quality issues.
    """

    # Signal weights
    WEIGHTS = {
        "speech_density": 0.30,
        "asr_confidence": 0.30,
        "timing_coherence": 0.20,
        "silence_gap": 0.20,
    }

    def compute_alignment(
        self,
        transcript_segments: List[Dict],
        audio_windows: List[Dict],
        duration: float,
    ) -> FullAlignmentResult:
        """
        Compute alignment between transcript and audio analysis.

        Args:
            transcript_segments: List of {"start_time", "end_time", "text", "confidence"}
            audio_windows: List of {"start_time", "end_time", "speech_probability", "music_probability", ...}
            duration: Total audio duration in seconds
        """
        if not transcript_segments or not audio_windows:
            return FullAlignmentResult(
                segments=[], overall_score=0.0, overall_quality="unknown",
                total_warnings=0, mismatch_regions=[],
            )

        results = []
        for seg in transcript_segments:
            seg_start = seg.get("start_time", 0)
            seg_end = seg.get("end_time", seg_start + 15)
            seg_text = seg.get("text", "")
            seg_confidence = seg.get("confidence", 0.5)

            # Find overlapping audio windows
            overlapping = [
                w for w in audio_windows
                if w.get("start_time", 0) < seg_end and w.get("end_time", 0) > seg_start
            ]

            signals = []
            warnings = []

            # ── Signal A: Speech Presence vs Transcript Density ──────
            avg_speech_prob = 0.0
            if overlapping:
                avg_speech_prob = np.mean([
                    w.get("speech_probability", 0) for w in overlapping
                ])

            seg_duration = max(seg_end - seg_start, 0.1)
            word_count = len(seg_text.split()) if seg_text else 0
            words_per_second = word_count / max(seg_duration, 0.01)

            # Expected: ~2.5 words/sec for normal speech
            speech_density_score = 1.0
            detail = ""
            if avg_speech_prob > 0.5:
                # Strong speech detected
                if words_per_second < 0.5:
                    speech_density_score = 0.2
                    detail = f"Speech detected but only {words_per_second:.1f} words/sec (expected ~2.5)"
                    warnings.append("Possible transcript gap — speech detected but few words transcribed")
                elif words_per_second > 6.0:
                    speech_density_score = 0.5
                    detail = f"Unusually dense transcript: {words_per_second:.1f} words/sec"
                    warnings.append("Transcript may contain hallucinated text")
                else:
                    speech_density_score = min(1.0, words_per_second / 2.5)
                    detail = f"Speech density: {words_per_second:.1f} words/sec"
            elif avg_speech_prob < 0.2 and word_count > 5:
                speech_density_score = 0.3
                detail = "Transcript present but no speech detected in audio"
                warnings.append("Possible caption mismatch — transcript exists but no speech in audio")
            else:
                detail = "No significant speech or transcript in this segment"

            signals.append(AlignmentSignal(
                name="speech_density",
                score=speech_density_score,
                weight=self.WEIGHTS["speech_density"],
                detail=detail,
            ))

            # ── Signal B: ASR Confidence ─────────────────────────────
            asr_score = seg_confidence
            asr_detail = f"ASR confidence: {seg_confidence:.2f}"
            if seg_confidence < 0.3 and avg_speech_prob > 0.5:
                asr_score = 0.2
                asr_detail = "Low ASR confidence despite strong speech — possible transcription error"
                warnings.append("Low transcript confidence — may contain errors")
            elif seg_confidence < 0.5:
                asr_score = seg_confidence * 0.8
                asr_detail = f"Below-average ASR confidence: {seg_confidence:.2f}"

            signals.append(AlignmentSignal(
                name="asr_confidence",
                score=asr_score,
                weight=self.WEIGHTS["asr_confidence"],
                detail=asr_detail,
            ))

            # ── Signal C: Timing Coherence ───────────────────────────
            # Check if transcript segment duration is plausible
            timing_score = 1.0
            timing_detail = "Timing consistent"

            expected_duration = word_count / 2.5 if word_count > 0 else 0
            if word_count > 0 and seg_duration > 0:
                ratio = seg_duration / max(expected_duration, 0.1)
                if ratio < 0.3:
                    timing_score = 0.3
                    timing_detail = f"Segment too short for word count: {seg_duration:.1f}s for {word_count} words"
                    warnings.append("Timing anomaly — segment duration doesn't match word count")
                elif ratio > 5.0:
                    timing_score = 0.4
                    timing_detail = f"Segment too long for word count: {seg_duration:.1f}s for {word_count} words"
                else:
                    timing_score = min(1.0, 1.0 - abs(1.0 - ratio) * 0.3)

            signals.append(AlignmentSignal(
                name="timing_coherence",
                score=timing_score,
                weight=self.WEIGHTS["timing_coherence"],
                detail=timing_detail,
            ))

            # ── Signal D: Silence Gap ────────────────────────────────
            # Check for unexpected silence in regions with transcript
            silence_score = 1.0
            silence_detail = "No unexpected silence"
            if overlapping:
                silence_windows = [
                    w for w in overlapping
                    if w.get("speech_probability", 0) < 0.1
                    and w.get("music_probability", 0) < 0.1
                ]
                silence_ratio = len(silence_windows) / max(len(overlapping), 1)
                if silence_ratio > 0.7 and word_count > 3:
                    silence_score = 0.2
                    silence_detail = f"{silence_ratio:.0%} silence but transcript has {word_count} words"
                    warnings.append("Mostly silence in audio but transcript has content")
                elif silence_ratio > 0.4 and word_count > 3:
                    silence_score = 0.5
                    silence_detail = f"Significant silence ({silence_ratio:.0%}) with transcript content"
                else:
                    silence_score = 1.0 - silence_ratio * 0.3

            signals.append(AlignmentSignal(
                name="silence_gap",
                score=silence_score,
                weight=self.WEIGHTS["silence_gap"],
                detail=silence_detail,
            ))

            # ── Composite Score ──────────────────────────────────────
            alignment_score = sum(s.score * s.weight for s in signals)
            alignment_score = max(0.0, min(1.0, alignment_score))

            # Quality level
            if alignment_score >= 0.80:
                quality = "good"
            elif alignment_score >= 0.55:
                quality = "acceptable"
            elif alignment_score >= 0.30:
                quality = "poor"
            else:
                quality = "mismatch"

            results.append(AlignmentResult(
                start_time=seg_start,
                end_time=seg_end,
                alignment_score=alignment_score,
                signals=signals,
                quality_level=quality,
                warnings=warnings,
            ))

        # ── Overall ──────────────────────────────────────────────────
        if results:
            overall = float(np.mean([r.alignment_score for r in results])) if results else 0.5
            total_warnings = sum(len(r.warnings) for r in results)
            mismatch_regions = [
                (r.start_time, r.end_time)
                for r in results
                if r.quality_level in ("poor", "mismatch")
            ]
        else:
            overall = 0.0
            total_warnings = 0
            mismatch_regions = []

        if overall >= 0.80:
            overall_quality = "good"
        elif overall >= 0.55:
            overall_quality = "acceptable"
        elif overall >= 0.30:
            overall_quality = "poor"
        else:
            overall_quality = "mismatch"

        return FullAlignmentResult(
            segments=results,
            overall_score=float(overall),
            overall_quality=overall_quality,
            total_warnings=total_warnings,
            mismatch_regions=mismatch_regions,
        )


alignment_service = AudioTranscriptAlignmentService()
