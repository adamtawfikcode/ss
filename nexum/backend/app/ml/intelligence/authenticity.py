"""
Nexum Intelligence Layer 3 — Authenticity & Manipulation Signals.

Probabilistic detection of media manipulation:
  Video: frame interpolation, CGI vs real, compression artifacts, lip-sync, deepfake
  Audio: formant drift, synthetic speech, over-denoising, time-stretch, voice cloning

All outputs are probabilistic with confidence bands — never binary truth.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceBand:
    """Probabilistic score with uncertainty range."""
    value: float
    lower: float
    upper: float
    method: str = "heuristic"

    @staticmethod
    def from_value(v: float, uncertainty: float = 0.15, method: str = "heuristic") -> "ConfidenceBand":
        return ConfidenceBand(
            value=round(v, 4),
            lower=round(max(0, v - uncertainty), 4),
            upper=round(min(1, v + uncertainty), 4),
            method=method,
        )


# ═══════════════════════════════════════════════════════════════════════
# Video Authenticity
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class VideoAuthenticityReport:
    """Comprehensive video authenticity analysis."""
    video_id: str
    frame_interpolation: ConfidenceBand
    cgi_probability: ConfidenceBand
    compression_artifact_score: ConfidenceBand
    lip_sync_mismatch: ConfidenceBand
    deepfake_probability: ConfidenceBand
    overall_authenticity: float  # 0=likely manipulated, 1=likely authentic
    flags: List[str] = field(default_factory=list)  # human-readable warnings
    confidence: float = 0.5


@dataclass
class AudioAuthenticityReport:
    """Comprehensive audio authenticity analysis."""
    video_id: str
    formant_drift_anomaly: ConfidenceBand
    synthetic_speech_likelihood: ConfidenceBand
    over_denoising_artifact: ConfidenceBand
    time_stretch_distortion: ConfidenceBand
    voice_cloning_likelihood: ConfidenceBand
    overall_authenticity: float
    flags: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class IntegrityReport:
    """Combined video + audio integrity assessment."""
    video_id: str
    video_report: VideoAuthenticityReport
    audio_report: AudioAuthenticityReport
    cross_modal_consistency: float  # do video and audio agree?
    overall_integrity_score: float  # 0=suspicious, 1=trustworthy
    risk_level: str                 # "low" | "medium" | "high" | "critical"
    confidence: float = 0.5


class AuthenticityService:
    """
    Detects manipulation signals in video and audio content.

    Uses heuristic spectral analysis, temporal consistency checks,
    and statistical anomaly detection. GPU-accelerated models can be
    plugged in for production deepfake detection.
    """

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("authenticity", True)
        logger.info("AuthenticityService initialized (enabled=%s)", self._enabled)

    def analyze_video(
        self, video_id: str, frames: List[Dict], audio_segments: List[Dict],
        transcript_segments: List[Dict] = None, fps: float = 30.0,
    ) -> IntegrityReport:
        """Full integrity analysis of a video."""
        try:
            vid_report = self._analyze_video_frames(video_id, frames, fps)
            aud_report = self._analyze_audio(video_id, audio_segments)

            # Cross-modal consistency
            cross_modal = self._check_cross_modal(vid_report, aud_report, transcript_segments)

            overall = (
                vid_report.overall_authenticity * 0.35 +
                aud_report.overall_authenticity * 0.35 +
                cross_modal * 0.3
            )

            if overall > 0.8:
                risk = "low"
            elif overall > 0.5:
                risk = "medium"
            elif overall > 0.25:
                risk = "high"
            else:
                risk = "critical"

            return IntegrityReport(
                video_id=video_id,
                video_report=vid_report,
                audio_report=aud_report,
                cross_modal_consistency=round(cross_modal, 3),
                overall_integrity_score=round(overall, 3),
                risk_level=risk,
                confidence=min(vid_report.confidence, aud_report.confidence),
            )
        except Exception as _exc:
            logger.warning(f"AuthenticityService.analyze_video failed: {_exc}")
            return None

    def _analyze_video_frames(
        self, video_id: str, frames: List[Dict], fps: float
    ) -> VideoAuthenticityReport:
        """Analyze video frames for manipulation signals."""
        if not frames:
            return VideoAuthenticityReport(
                video_id=video_id,
                frame_interpolation=ConfidenceBand.from_value(0.1, 0.3),
                cgi_probability=ConfidenceBand.from_value(0.1, 0.3),
                compression_artifact_score=ConfidenceBand.from_value(0.2, 0.2),
                lip_sync_mismatch=ConfidenceBand.from_value(0.1, 0.3),
                deepfake_probability=ConfidenceBand.from_value(0.05, 0.2),
                overall_authenticity=0.85,
                confidence=0.15,
            )

        # Frame interpolation detection via scene change regularity
        scene_changes = [f for f in frames if f.get("is_scene_change")]
        if len(scene_changes) > 2:
            intervals = []
            timestamps = [f.get("timestamp", 0) for f in scene_changes]
            for i in range(1, len(timestamps)):
                intervals.append(timestamps[i] - timestamps[i - 1])
            if intervals:
                cv = float(np.std(intervals) / max(float(np.mean(intervals)), 0.01)) if len(intervals) > 1 else 0.0
                # Unnaturally regular intervals suggest interpolation
                interp_score = max(0, 1.0 - cv) * 0.5  # regular = suspicious
            else:
                interp_score = 0.1
        else:
            interp_score = 0.1

        # CGI detection heuristic: low OCR confidence + high visual tag count
        ocr_confs = [f.get("ocr_confidence", 0.5) for f in frames if f.get("ocr_confidence")]
        tag_counts = [len(f.get("visual_tags") or []) for f in frames]
        if ocr_confs and tag_counts:
            low_ocr = float(np.mean([1 if c < 0.3 else 0 for c in ocr_confs])) if ocr_confs else 0.0
            high_tags = float(np.mean([1 if t > 10 else 0 for t in tag_counts])) if tag_counts else 0.0
            cgi_score = (low_ocr * 0.3 + high_tags * 0.2) * 0.5
        else:
            cgi_score = 0.1

        # Compression artifacts from OCR quality degradation
        if ocr_confs:
            compression = max(0, 1.0 - float(np.mean(ocr_confs))) if ocr_confs else 0.3
        else:
            compression = 0.2

        # Lip sync: would need audio-visual alignment model, use placeholder
        lip_sync = 0.05  # default: assume good sync

        # Deepfake composite score
        deepfake = min(1.0, interp_score * 0.3 + cgi_score * 0.3 + lip_sync * 0.4)

        overall = 1.0 - (deepfake * 0.4 + compression * 0.2 + interp_score * 0.2 + cgi_score * 0.2)

        flags = []
        if interp_score > 0.6:
            flags.append("Unusually regular frame timing detected")
        if compression > 0.7:
            flags.append("Heavy compression artifacts")
        if deepfake > 0.5:
            flags.append("Elevated deepfake indicators")

        return VideoAuthenticityReport(
            video_id=video_id,
            frame_interpolation=ConfidenceBand.from_value(interp_score, 0.2),
            cgi_probability=ConfidenceBand.from_value(cgi_score, 0.25),
            compression_artifact_score=ConfidenceBand.from_value(compression, 0.15),
            lip_sync_mismatch=ConfidenceBand.from_value(lip_sync, 0.3),
            deepfake_probability=ConfidenceBand.from_value(deepfake, 0.2),
            overall_authenticity=round(max(0, min(1, overall)), 3),
            flags=flags,
            confidence=min(0.75, 0.2 + len(frames) * 0.002),
        )

    def _analyze_audio(
        self, video_id: str, audio_segments: List[Dict]
    ) -> AudioAuthenticityReport:
        """Analyze audio for synthesis and manipulation artifacts."""
        if not audio_segments:
            return AudioAuthenticityReport(
                video_id=video_id,
                formant_drift_anomaly=ConfidenceBand.from_value(0.05, 0.3),
                synthetic_speech_likelihood=ConfidenceBand.from_value(0.05, 0.3),
                over_denoising_artifact=ConfidenceBand.from_value(0.1, 0.2),
                time_stretch_distortion=ConfidenceBand.from_value(0.05, 0.3),
                voice_cloning_likelihood=ConfidenceBand.from_value(0.02, 0.2),
                overall_authenticity=0.9,
                confidence=0.15,
            )

        # Formant drift: check spectral centroid consistency
        centroids = [(s.get("spectral_centroid") or 2000) for s in audio_segments
                     if (s.get("speech_probability") or 0) > 0.5]
        if len(centroids) > 5:
            centroid_cv = float(np.std(centroids) / max(float(np.mean(centroids)), 1)) if len(centroids) > 1 else 0.0
            # Very low CV = unnaturally consistent = possible synthesis
            formant_drift = max(0, 0.5 - centroid_cv) * 2 if centroid_cv < 0.5 else 0.0
        else:
            formant_drift = 0.05

        # Over-denoising: very low dynamic range in speech segments
        dynamic_ranges = [(s.get("dynamic_range_db") or 15) for s in audio_segments
                          if (s.get("speech_probability") or 0) > 0.5]
        if dynamic_ranges:
            avg_dr = float(np.mean(dynamic_ranges)) if dynamic_ranges else 12.0
            over_denoise = max(0, (8 - avg_dr) / 8) if avg_dr < 8 else 0.0
        else:
            over_denoise = 0.1

        # Time stretch: check zero crossing rate anomalies
        zcrs = [(s.get("zero_crossing_rate") or 0.05) for s in audio_segments]
        if len(zcrs) > 5:
            zcr_cv = float(np.std(zcrs) / max(float(np.mean(zcrs)), 1e-6)) if len(zcrs) > 1 else 0.0
            time_stretch = max(0, zcr_cv - 1.5) * 0.3 if zcr_cv > 1.5 else 0.0
        else:
            time_stretch = 0.05

        # Synthetic speech: harmonic ratio consistency (synth = very consistent)
        harmonics = [(s.get("harmonic_ratio") or 0.5) for s in audio_segments
                     if (s.get("speech_probability") or 0) > 0.5]
        if len(harmonics) > 5:
            h_cv = float(np.std(harmonics) / max(float(np.mean(harmonics)), 1e-6)) if len(harmonics) > 1 else 0.0
            synthetic = max(0, (0.1 - h_cv) * 5) if h_cv < 0.1 else 0.0
        else:
            synthetic = 0.05

        # Voice cloning composite
        cloning = min(1.0, synthetic * 0.4 + formant_drift * 0.3 + over_denoise * 0.3)

        overall = 1.0 - (cloning * 0.3 + synthetic * 0.2 + over_denoise * 0.2 +
                         time_stretch * 0.15 + formant_drift * 0.15)

        flags = []
        if synthetic > 0.5:
            flags.append("Elevated synthetic speech indicators")
        if over_denoise > 0.6:
            flags.append("Possible over-denoising artifacts")
        if cloning > 0.4:
            flags.append("Voice cloning indicators present")

        return AudioAuthenticityReport(
            video_id=video_id,
            formant_drift_anomaly=ConfidenceBand.from_value(formant_drift, 0.2),
            synthetic_speech_likelihood=ConfidenceBand.from_value(synthetic, 0.2),
            over_denoising_artifact=ConfidenceBand.from_value(over_denoise, 0.15),
            time_stretch_distortion=ConfidenceBand.from_value(time_stretch, 0.25),
            voice_cloning_likelihood=ConfidenceBand.from_value(cloning, 0.2),
            overall_authenticity=round(max(0, min(1, overall)), 3),
            flags=flags,
            confidence=min(0.75, 0.2 + len(audio_segments) * 0.005),
        )

    def _check_cross_modal(
        self, vid: VideoAuthenticityReport, aud: AudioAuthenticityReport,
        transcript: List[Dict] = None,
    ) -> float:
        """Check consistency between video and audio authenticity signals."""
        vid_score = vid.overall_authenticity
        aud_score = aud.overall_authenticity
        # If both agree, consistency is high
        agreement = 1.0 - abs(vid_score - aud_score)
        return round(agreement * 0.7 + min(vid_score, aud_score) * 0.3, 3)
