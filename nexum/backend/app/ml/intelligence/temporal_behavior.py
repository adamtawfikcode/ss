"""
Nexum Intelligence Layer 1 — Temporal Behavior Intelligence.

Extracts time-series patterns from all timestamped data:
  - Channel-level: upload cadence, periodicity, break/spike anomalies, timezone inference
  - In-video: speech rate curves, energy curves, music density, silence distribution
  - Derived: excitement ramp, attention decay, highlight probability, info density

All outputs are probabilistic with confidence bands.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Classes — Channel Temporal
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class UploadCadenceFingerprint:
    """Statistical fingerprint of channel upload behavior."""
    channel_id: str
    mean_interval_hours: float
    std_interval_hours: float
    median_interval_hours: float
    weekday_distribution: List[float]  # 7 values, Mon=0
    hour_distribution: List[float]     # 24 values
    periodicity_score: float           # 0-1: how periodic
    dominant_period_days: Optional[float] = None
    confidence: float = 0.5

    @property
    def regularity_index(self) -> float:
        if self.mean_interval_hours == 0:
            return 0.0
        cv = self.std_interval_hours / max(self.mean_interval_hours, 1e-6)
        return max(0.0, 1.0 - min(cv, 2.0) / 2.0)


@dataclass
class SeasonalPattern:
    """Weekly and seasonal periodicity detection."""
    channel_id: str
    weekly_autocorrelation: float
    monthly_autocorrelation: float
    quarterly_autocorrelation: float
    peak_upload_day: int              # 0=Mon
    peak_upload_hour: int             # 0-23 UTC
    seasonal_strength: float          # 0-1
    confidence: float = 0.5


@dataclass
class BreakSpikeAnomaly:
    """Detects unusual gaps followed by activity spikes."""
    channel_id: str
    break_start: str                  # ISO datetime
    break_end: str
    break_duration_days: float
    post_break_upload_rate: float     # uploads/day in 7-day window after
    pre_break_upload_rate: float
    spike_ratio: float                # post/pre rate ratio
    anomaly_score: float              # 0-1
    confidence: float = 0.5


@dataclass
class TimezoneInference:
    """Inferred creator timezone from posting patterns."""
    channel_id: str
    inferred_tz_offset: float         # hours from UTC
    inferred_tz_name: Optional[str]
    peak_local_hour: int
    confidence: float = 0.5
    method: str = "posting_pattern"


@dataclass
class GrowthCurve:
    """Channel growth acceleration metrics."""
    channel_id: str
    growth_rate_30d: float            # subscriber growth rate
    growth_rate_90d: float
    acceleration: float               # second derivative
    inflection_detected: bool
    projected_30d: Optional[float] = None
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# Data Classes — In-Video Temporal Curves
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class TemporalCurve:
    """Generic time-series curve with regular sampling."""
    video_id: str
    curve_type: str                   # "speech_rate" | "energy" | "music_density" | etc.
    sample_rate_hz: float             # samples per second
    values: List[float]
    min_val: float = 0.0
    max_val: float = 1.0
    mean_val: float = 0.0
    std_val: float = 0.0

    def at_time(self, seconds: float) -> float:
        idx = int(seconds * self.sample_rate_hz)
        if 0 <= idx < len(self.values):
            return self.values[idx]
        return self.mean_val

    def window_stats(self, start_s: float, end_s: float) -> Dict:
        i0 = max(0, int(start_s * self.sample_rate_hz))
        i1 = min(len(self.values), int(end_s * self.sample_rate_hz))
        window = self.values[i0:i1]
        if not window:
            return {"mean": 0, "max": 0, "min": 0, "std": 0}
        arr = np.array(window)
        return {"mean": float(arr.mean()), "max": float(arr.max()),
                "min": float(arr.min()), "std": float(arr.std())}


@dataclass
class SilenceDistribution:
    """Silence gap analysis within a video."""
    video_id: str
    total_silence_seconds: float
    silence_ratio: float              # silence/total duration
    silence_gaps: List[Tuple[float, float]]  # (start, end) pairs
    mean_gap_duration: float
    max_gap_duration: float
    gap_count: int
    dramatic_pause_count: int         # gaps 1-3s before speech


@dataclass
class ScenePacingRhythm:
    """Scene transition pacing analysis."""
    video_id: str
    scene_count: int
    mean_scene_duration: float
    std_scene_duration: float
    pacing_curve: List[float]         # scene durations in order
    acceleration_index: float         # negative = scenes get shorter (faster pace)
    rhythm_regularity: float          # 0-1: how evenly spaced


# ═══════════════════════════════════════════════════════════════════════
# Data Classes — Derived Metrics
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DerivedTemporalMetrics:
    """High-level interpretive metrics derived from temporal curves."""
    video_id: str
    excitement_ramp_score: float      # 0-1: does energy build to climax?
    attention_decay_probability: float # 0-1: likelihood audience drops off
    highlight_timestamps: List[Tuple[float, float]]  # (timestamp, probability)
    info_density_per_minute: List[float]  # words + entities + visuals per minute
    engagement_spike_timestamps: List[float]
    pacing_quality_score: float       # 0-1: well-paced?
    confidence: float = 0.5


# ═══════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════


class TemporalBehaviorService:
    """
    Extracts temporal intelligence from channels and videos.

    Channel-level: operates on lists of upload timestamps + engagement metrics.
    Video-level: operates on audio segments, transcript segments, frames.
    """

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("temporal_behavior", True)
        logger.info("TemporalBehaviorService initialized (enabled=%s)", self._enabled)

    # ── Channel-Level Analysis ───────────────────────────────────────

    def compute_upload_cadence(
        self, channel_id: str, upload_timestamps: List[float]
    ) -> UploadCadenceFingerprint:
        """Compute upload cadence fingerprint from sorted Unix timestamps."""
        try:
            if len(upload_timestamps) < 3:
                return UploadCadenceFingerprint(
                    channel_id=channel_id, mean_interval_hours=0,
                    std_interval_hours=0, median_interval_hours=0,
                    weekday_distribution=[1 / 7] * 7, hour_distribution=[1 / 24] * 24,
                    periodicity_score=0, confidence=0.1,
                )

            ts = sorted(upload_timestamps)
            intervals = np.diff(ts) / 3600.0  # hours

            # Weekday and hour distributions
            from datetime import datetime, timezone
            dts = [datetime.fromtimestamp(t, tz=timezone.utc) for t in ts]
            wd = np.zeros(7)
            hd = np.zeros(24)
            for dt in dts:
                wd[dt.weekday()] += 1
                hd[dt.hour] += 1
            wd = wd / max(wd.sum(), 1)
            hd = hd / max(hd.sum(), 1)

            # Periodicity via autocorrelation
            if len(intervals) > 14:
                norm_intervals = (intervals - intervals.mean()) / max(intervals.std(), 1e-6)
                acf_7 = float(np.mean(norm_intervals[:-7] * norm_intervals[7:]))
                periodicity = max(0.0, min(1.0, abs(acf_7)))
            else:
                periodicity = 0.0

            return UploadCadenceFingerprint(
                channel_id=channel_id,
                mean_interval_hours=float(intervals.mean()),
                std_interval_hours=float(intervals.std()),
                median_interval_hours=float(np.median(intervals)),
                weekday_distribution=wd.tolist(),
                hour_distribution=hd.tolist(),
                periodicity_score=periodicity,
                dominant_period_days=float(np.median(intervals) / 24.0) if len(intervals) > 2 else None,
                confidence=min(0.95, 0.3 + len(ts) * 0.02),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_upload_cadence failed: {_exc}")
            return None

    def detect_break_spike_anomalies(
        self, channel_id: str, upload_timestamps: List[float], threshold_std: float = 2.0
    ) -> List[BreakSpikeAnomaly]:
        """Detect unusual gaps followed by upload bursts."""
        try:
            if len(upload_timestamps) < 5:
                return []

            ts = sorted(upload_timestamps)
            intervals = np.diff(ts) / 86400.0  # days
            mean_int = float(intervals.mean())
            std_int = float(intervals.std()) if len(intervals) > 2 else mean_int

            anomalies = []
            for i, gap in enumerate(intervals):
                if gap > mean_int + threshold_std * max(std_int, 1):
                    # Check post-break rate (7-day window)
                    break_end = ts[i + 1]
                    post_window = [t for t in ts if break_end <= t <= break_end + 7 * 86400]
                    pre_window = [t for t in ts if ts[i] - 7 * 86400 <= t <= ts[i]]
                    post_rate = len(post_window) / 7.0
                    pre_rate = len(pre_window) / 7.0

                    from datetime import datetime, timezone
                    anomalies.append(BreakSpikeAnomaly(
                        channel_id=channel_id,
                        break_start=datetime.fromtimestamp(ts[i], tz=timezone.utc).isoformat(),
                        break_end=datetime.fromtimestamp(break_end, tz=timezone.utc).isoformat(),
                        break_duration_days=float(gap),
                        post_break_upload_rate=post_rate,
                        pre_break_upload_rate=pre_rate,
                        spike_ratio=post_rate / max(pre_rate, 0.01),
                        anomaly_score=min(1.0, (gap - mean_int) / max(std_int * 3, 1)),
                        confidence=0.6,
                    ))
            return anomalies
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.detect_break_spike_anomalies failed: {_exc}")
            return None

    def infer_timezone(
        self, channel_id: str, upload_timestamps: List[float]
    ) -> TimezoneInference:
        """Infer creator timezone from upload hour clustering."""
        try:
            from datetime import datetime, timezone
            if len(upload_timestamps) < 10:
                return TimezoneInference(
                    channel_id=channel_id, inferred_tz_offset=0,
                    inferred_tz_name="UTC", peak_local_hour=12, confidence=0.1,
                )

            hours_utc = np.array([datetime.fromtimestamp(t, tz=timezone.utc).hour +
                                  datetime.fromtimestamp(t, tz=timezone.utc).minute / 60
                                  for t in upload_timestamps])

            # Find offset that puts peak in 9-17 local time (business hours)
            best_offset = 0
            best_score = -1
            for offset in range(-12, 13):
                local_hours = (hours_utc + offset) % 24
                business_ratio = np.sum((local_hours >= 8) & (local_hours <= 20)) / len(local_hours)
                if business_ratio > best_score:
                    best_score = business_ratio
                    best_offset = offset

            local_hours = (hours_utc + best_offset) % 24
            peak_hour = int(np.argmax(np.bincount(local_hours.astype(int), minlength=24)))

            tz_names = {
                -8: "PST", -7: "MST", -6: "CST", -5: "EST",
                0: "UTC", 1: "CET", 2: "EET", 3: "AST",
                5.5: "IST", 8: "CST_CN", 9: "JST",
            }

            return TimezoneInference(
                channel_id=channel_id,
                inferred_tz_offset=float(best_offset),
                inferred_tz_name=tz_names.get(best_offset, f"UTC{best_offset:+.0f}"),
                peak_local_hour=peak_hour,
                confidence=min(0.9, best_score),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.infer_timezone failed: {_exc}")
            return None

    # ── In-Video Temporal Analysis ───────────────────────────────────

    def compute_speech_rate_curve(
        self, video_id: str, segments: List[Dict], duration_s: float, window_s: float = 5.0
    ) -> TemporalCurve:
        """Compute words-per-second over time from transcript segments."""
        try:
            window_s = max(window_s, 0.01)
            sample_rate = 1.0 / window_s
            n_windows = max(1, int(duration_s / window_s))
            values = [0.0] * n_windows

            for seg in segments:
                start = seg.get("start_time", 0)
                end = seg.get("end_time", start + 1)
                text = seg.get("text") or ""
                words = len(text.split())
                dur = max(end - start, 0.1)
                wps = words / dur

                i0 = int(start / window_s)
                i1 = min(int(end / window_s) + 1, n_windows)
                for i in range(i0, i1):
                    values[i] = max(values[i], wps)

            arr = np.array(values)
            return TemporalCurve(
                video_id=video_id, curve_type="speech_rate",
                sample_rate_hz=sample_rate, values=values,
                min_val=float(arr.min()), max_val=float(arr.max()),
                mean_val=float(arr.mean()), std_val=float(arr.std()),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_speech_rate_curve failed: {_exc}")
            return None

    def compute_energy_curve(
        self, video_id: str, audio_segments: List[Dict], duration_s: float, window_s: float = 2.0
    ) -> TemporalCurve:
        """Compute audio energy intensity from audio segment data."""
        try:
            window_s = max(window_s, 0.01)
            sample_rate = 1.0 / window_s
            n_windows = max(1, int(duration_s / window_s))
            values = [0.0] * n_windows

            for seg in audio_segments:
                start = seg.get("start_time", 0)
                end = seg.get("end_time", start + window_s)
                loudness = seg.get("loudness_lufs")
                if loudness is not None:
                    energy = max(0.0, min(1.0, (loudness + 60) / 60))
                else:
                    energy = seg.get("speech_probability", 0.3)

                i0 = int(start / window_s)
                i1 = min(int(end / window_s) + 1, n_windows)
                for i in range(i0, i1):
                    values[i] = max(values[i], energy)

            arr = np.array(values)
            return TemporalCurve(
                video_id=video_id, curve_type="energy",
                sample_rate_hz=sample_rate, values=values,
                min_val=float(arr.min()), max_val=float(arr.max()),
                mean_val=float(arr.mean()), std_val=float(arr.std()),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_energy_curve failed: {_exc}")
            return None

    def compute_music_density_curve(
        self, video_id: str, audio_segments: List[Dict], duration_s: float, window_s: float = 5.0
    ) -> TemporalCurve:
        """Music probability density over time."""
        try:
            window_s = max(window_s, 0.01)
            sample_rate = 1.0 / window_s
            n_windows = max(1, int(duration_s / window_s))
            values = [0.0] * n_windows

            for seg in audio_segments:
                start = seg.get("start_time", 0)
                end = seg.get("end_time", start + window_s)
                music_prob = seg.get("music_probability", 0)
                i0 = int(start / window_s)
                i1 = min(int(end / window_s) + 1, n_windows)
                for i in range(i0, i1):
                    values[i] = max(values[i], music_prob)

            arr = np.array(values)
            return TemporalCurve(
                video_id=video_id, curve_type="music_density",
                sample_rate_hz=sample_rate, values=values,
                min_val=float(arr.min()), max_val=float(arr.max()),
                mean_val=float(arr.mean()), std_val=float(arr.std()),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_music_density_curve failed: {_exc}")
            return None

    def compute_silence_distribution(
        self, video_id: str, audio_segments: List[Dict], duration_s: float,
        silence_threshold: float = 0.1
    ) -> SilenceDistribution:
        """Analyze silence gaps in audio timeline."""
        try:
            sorted_segs = sorted(audio_segments, key=lambda s: s.get("start_time", 0))
            gaps = []
            prev_end = 0.0

            for seg in sorted_segs:
                start = seg.get("start_time", 0)
                speech_prob = seg.get("speech_probability", 0.5)
                if start > prev_end + 0.5 and speech_prob < silence_threshold:
                    gaps.append((prev_end, start))
                prev_end = max(prev_end, seg.get("end_time", start + 1))

            gap_durations = [e - s for s, e in gaps]
            total_silence = sum(gap_durations)
            dramatic = sum(1 for d in gap_durations if 0.8 <= d <= 3.5)

            return SilenceDistribution(
                video_id=video_id,
                total_silence_seconds=total_silence,
                silence_ratio=total_silence / max(duration_s, 1),
                silence_gaps=gaps,
                mean_gap_duration=float(np.mean(gap_durations)) if gap_durations else 0,
                max_gap_duration=max(gap_durations, default=0),
                gap_count=len(gaps),
                dramatic_pause_count=dramatic,
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_silence_distribution failed: {_exc}")
            return None

    # ── Derived Metrics ──────────────────────────────────────────────

    def compute_derived_metrics(
        self, video_id: str, speech_curve: TemporalCurve,
        energy_curve: TemporalCurve, music_curve: TemporalCurve,
        duration_s: float, segment_word_counts: List[Tuple[float, int]] = None,
    ) -> DerivedTemporalMetrics:
        """Compute high-level interpretive metrics from temporal curves."""
        # Guard against None curves
        _empty = TemporalCurve(video_id=video_id, curve_type="empty", sample_rate_hz=1.0, values=[0.0], min_val=0, max_val=0, mean_val=0, std_val=0)
        if speech_curve is None:
            speech_curve = _empty
        if energy_curve is None:
            energy_curve = _empty
        if music_curve is None:
            music_curve = _empty
        duration_s = max(duration_s, 0.01)

        # Excitement ramp: is the last quarter more energetic than first?
        try:
            n = len(energy_curve.values)
            if n > 4:
                q1 = np.mean(energy_curve.values[:n // 4]) if n >= 4 else 0.5
                q4 = np.mean(energy_curve.values[3 * n // 4:]) if n >= 4 else 0.5
                excitement_ramp = max(0.0, min(1.0, (q4 - q1) / max(q1, 0.01) * 0.5 + 0.5))
            else:
                excitement_ramp = 0.5

            # Attention decay: does speech rate drop in second half?
            sn = len(speech_curve.values)
            if sn > 4:
                first_half = np.mean(speech_curve.values[:sn // 2]) if sn >= 2 else 0.5
                second_half = np.mean(speech_curve.values[sn // 2:]) if sn >= 2 else 0.5
                decay = max(0.0, min(1.0, (first_half - second_half) / max(first_half, 0.01)))
            else:
                decay = 0.3

            # Highlight detection: peaks in combined energy+speech
            highlights = []
            if n > 10:
                combined = np.array(energy_curve.values[:min(n, len(speech_curve.values))])
                if len(speech_curve.values) >= len(combined):
                    combined = combined + np.array(speech_curve.values[:len(combined)])
                threshold = np.percentile(combined, 85)
                for i, v in enumerate(combined):
                    if v > threshold:
                        ts = i / max(energy_curve.sample_rate_hz, 0.01)
                        prob = min(1.0, (v - threshold) / max(threshold, 0.01))
                        highlights.append((ts, round(prob, 3)))

            # Info density per minute
            info_density = []
            minutes = max(1, int(duration_s / 60))
            for m in range(minutes):
                speech_stats = speech_curve.window_stats(m * 60, (m + 1) * 60)
                energy_stats = energy_curve.window_stats(m * 60, (m + 1) * 60)
                density = speech_stats["mean"] * 30 + energy_stats["mean"] * 20
                info_density.append(round(density, 1))

            # Pacing quality: moderate variance in energy is good
            if energy_curve.std_val > 0:
                cv = energy_curve.std_val / max(energy_curve.mean_val, 0.01)
                pacing = max(0.0, min(1.0, 1.0 - abs(cv - 0.4) * 2))
            else:
                pacing = 0.3

            return DerivedTemporalMetrics(
                video_id=video_id,
                excitement_ramp_score=round(excitement_ramp, 3),
                attention_decay_probability=round(decay, 3),
                highlight_timestamps=highlights[:20],
                info_density_per_minute=info_density,
                engagement_spike_timestamps=[h[0] for h in highlights[:10]],
                pacing_quality_score=round(pacing, 3),
                confidence=min(0.85, 0.4 + len(energy_curve.values) * 0.001),
            )
        except Exception as _exc:
            logger.warning(f"TemporalBehaviorService.compute_derived_metrics failed: {_exc}")
            return None
