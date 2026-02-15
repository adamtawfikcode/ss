"""
Nexum Acoustic Change Point Detection.

Detects timestamps where the acoustic distribution shifts significantly:
  - silence → music
  - music → speech
  - calm → explosion
  - monologue → applause
  - beat drop / music transition

Detection Methods:
  1. Spectral Distance — cosine distance between adjacent mel-spectrogram windows
  2. Event Class Delta — probability distribution shift across AudioSet classes
  3. Energy Spike — RMS energy ratio between adjacent windows
  4. Harmonic Shift — change in harmonic-percussive balance

Output: list of AcousticChangePoint with timestamp, magnitude, and transition type.

Enables:
  - Highlight detection (beat drops, crowd reactions, dramatic shifts)
  - Better timeline UI (sound scene segmentation)
  - Query anchoring ("moment when music stops")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TransitionType(str, Enum):
    SILENCE_TO_MUSIC = "silence_to_music"
    MUSIC_TO_SILENCE = "music_to_silence"
    SILENCE_TO_SPEECH = "silence_to_speech"
    SPEECH_TO_SILENCE = "speech_to_silence"
    MUSIC_TO_SPEECH = "music_to_speech"
    SPEECH_TO_MUSIC = "speech_to_music"
    ENERGY_SPIKE = "energy_spike"
    ENERGY_DROP = "energy_drop"
    EVENT_ONSET = "event_onset"        # laughter, applause, explosion, etc.
    HARMONIC_SHIFT = "harmonic_shift"  # tonal → percussive or vice versa
    TEMPO_CHANGE = "tempo_change"
    SPECTRAL_SHIFT = "spectral_shift"  # general timbral change
    UNKNOWN = "unknown"


@dataclass
class AcousticChangePoint:
    """A detected point of acoustic change."""
    timestamp: float               # seconds
    magnitude: float              # 0–1, how dramatic the change is
    transition_type: TransitionType
    detail: str = ""              # human-readable description
    from_state: str = ""          # e.g. "music", "speech", "silence"
    to_state: str = ""            # e.g. "applause", "silence"
    spectral_distance: float = 0.0
    energy_ratio: float = 0.0
    event_delta: float = 0.0
    harmonic_shift: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "timestamp": round(self.timestamp, 2),
            "magnitude": round(self.magnitude, 3),
            "transition_type": self.transition_type.value,
            "detail": self.detail,
            "from_state": self.from_state,
            "to_state": self.to_state,
        }


@dataclass
class ChangePointResult:
    """All change points detected in an audio file."""
    change_points: List[AcousticChangePoint]
    total_duration: float
    num_scenes: int              # number of distinct acoustic scenes
    dominant_transitions: List[str]  # most common transition types

    def to_dict(self) -> Dict:
        return {
            "total_change_points": len(self.change_points),
            "total_duration": round(self.total_duration, 1),
            "num_scenes": self.num_scenes,
            "dominant_transitions": self.dominant_transitions,
            "change_points": [cp.to_dict() for cp in self.change_points],
        }


class ChangePointDetector:
    """
    Multi-signal acoustic change point detection.

    Takes pre-computed audio window results and finds transition boundaries.
    """

    # Thresholds
    SPECTRAL_THRESHOLD = 0.35
    ENERGY_RATIO_THRESHOLD = 3.0
    EVENT_DELTA_THRESHOLD = 0.40
    HARMONIC_SHIFT_THRESHOLD = 0.30
    MIN_MAGNITUDE = 0.25  # suppress weak change points

    def detect(self, audio_windows: List[Dict]) -> ChangePointResult:
        """
        Detect acoustic change points from pre-analyzed audio windows.

        Each window dict should contain:
            start_time, end_time, music_probability, speech_probability,
            dominant_source, event_tags, loudness_lufs, harmonic_ratio,
            spectral_centroid, bpm
        """
        if len(audio_windows) < 2:
            return ChangePointResult(
                change_points=[], total_duration=0, num_scenes=1,
                dominant_transitions=[],
            )

        change_points: List[AcousticChangePoint] = []

        for i in range(1, len(audio_windows)):
            prev = audio_windows[i - 1]
            curr = audio_windows[i]

            timestamp = curr.get("start_time", 0)
            signals = []

            # ── Signal 1: Source Class Delta ──────────────────────────
            source_delta = self._compute_source_delta(prev, curr)
            if source_delta["magnitude"] > self.EVENT_DELTA_THRESHOLD:
                signals.append(source_delta)

            # ── Signal 2: Energy Change ──────────────────────────────
            energy_signal = self._compute_energy_change(prev, curr)
            if energy_signal["magnitude"] > 0.3:
                signals.append(energy_signal)

            # ── Signal 3: Spectral Shift ─────────────────────────────
            spectral_signal = self._compute_spectral_shift(prev, curr)
            if spectral_signal["magnitude"] > self.SPECTRAL_THRESHOLD:
                signals.append(spectral_signal)

            # ── Signal 4: Harmonic Shift ─────────────────────────────
            harmonic_signal = self._compute_harmonic_shift(prev, curr)
            if harmonic_signal["magnitude"] > self.HARMONIC_SHIFT_THRESHOLD:
                signals.append(harmonic_signal)

            # ── Signal 5: Event Onset ────────────────────────────────
            event_signal = self._detect_event_onset(prev, curr)
            if event_signal and event_signal["magnitude"] > 0.3:
                signals.append(event_signal)

            # ── Aggregate ────────────────────────────────────────────
            if signals:
                # Weighted max — the strongest signal dominates
                best = max(signals, key=lambda s: s["magnitude"])
                magnitude = best["magnitude"]

                # Boost if multiple signals agree
                if len(signals) >= 2:
                    magnitude = min(1.0, magnitude * 1.2)
                if len(signals) >= 3:
                    magnitude = min(1.0, magnitude * 1.15)

                if magnitude >= self.MIN_MAGNITUDE:
                    cp = AcousticChangePoint(
                        timestamp=timestamp,
                        magnitude=magnitude,
                        transition_type=best["type"],
                        detail=best.get("detail", ""),
                        from_state=best.get("from_state", ""),
                        to_state=best.get("to_state", ""),
                        spectral_distance=spectral_signal.get("magnitude", 0),
                        energy_ratio=energy_signal.get("raw_ratio", 1.0),
                        event_delta=source_delta.get("magnitude", 0),
                        harmonic_shift=harmonic_signal.get("magnitude", 0),
                    )
                    change_points.append(cp)

        # Suppress duplicates within 3 seconds (keep stronger)
        change_points = self._suppress_nearby(change_points, min_gap=3.0)

        # Compute scene count
        num_scenes = len(change_points) + 1

        # Dominant transitions
        transition_counts: Dict[str, int] = {}
        for cp in change_points:
            t = cp.transition_type.value
            transition_counts[t] = transition_counts.get(t, 0) + 1
        dominant = sorted(transition_counts, key=transition_counts.get, reverse=True)[:3]

        duration = audio_windows[-1].get("end_time", 0) if audio_windows else 0

        return ChangePointResult(
            change_points=change_points,
            total_duration=duration,
            num_scenes=num_scenes,
            dominant_transitions=dominant,
        )

    # ── Signal Computation ───────────────────────────────────────────────

    def _compute_source_delta(self, prev: Dict, curr: Dict) -> Dict:
        """Detect shift in dominant audio source (music/speech/silence)."""
        prev_music = prev.get("music_probability", 0)
        prev_speech = prev.get("speech_probability", 0)
        curr_music = curr.get("music_probability", 0)
        curr_speech = curr.get("speech_probability", 0)

        prev_source = prev.get("dominant_source", "unknown")
        curr_source = curr.get("dominant_source", "unknown")

        music_delta = abs(curr_music - prev_music)
        speech_delta = abs(curr_speech - prev_speech)
        magnitude = max(music_delta, speech_delta)

        transition_type = TransitionType.UNKNOWN
        from_state = prev_source
        to_state = curr_source
        detail = ""

        if prev_source != curr_source and magnitude > 0.3:
            key = f"{prev_source}_to_{curr_source}"
            type_map = {
                "silence_to_music": TransitionType.SILENCE_TO_MUSIC,
                "music_to_silence": TransitionType.MUSIC_TO_SILENCE,
                "silence_to_speech": TransitionType.SILENCE_TO_SPEECH,
                "speech_to_silence": TransitionType.SPEECH_TO_SILENCE,
                "music_to_speech": TransitionType.MUSIC_TO_SPEECH,
                "speech_to_music": TransitionType.SPEECH_TO_MUSIC,
            }
            transition_type = type_map.get(key, TransitionType.SPECTRAL_SHIFT)
            detail = f"Source transition: {prev_source} → {curr_source}"

        return {
            "magnitude": magnitude,
            "type": transition_type,
            "from_state": from_state,
            "to_state": to_state,
            "detail": detail,
        }

    def _compute_energy_change(self, prev: Dict, curr: Dict) -> Dict:
        """Detect energy spikes or drops via loudness."""
        prev_loud = prev.get("loudness_lufs", -30)
        curr_loud = curr.get("loudness_lufs", -30)

        # Convert LUFS to linear for ratio
        prev_lin = 10 ** (prev_loud / 20) + 1e-8
        curr_lin = 10 ** (curr_loud / 20) + 1e-8
        ratio = curr_lin / prev_lin
        inv_ratio = prev_lin / curr_lin

        max_ratio = max(ratio, inv_ratio)
        magnitude = min(1.0, (max_ratio - 1.0) / 5.0)  # normalize

        if ratio > self.ENERGY_RATIO_THRESHOLD:
            return {
                "magnitude": magnitude,
                "type": TransitionType.ENERGY_SPIKE,
                "detail": f"Energy spike: {curr_loud:.0f} LUFS (from {prev_loud:.0f})",
                "from_state": f"{prev_loud:.0f} LUFS",
                "to_state": f"{curr_loud:.0f} LUFS",
                "raw_ratio": ratio,
            }
        elif inv_ratio > self.ENERGY_RATIO_THRESHOLD:
            return {
                "magnitude": magnitude,
                "type": TransitionType.ENERGY_DROP,
                "detail": f"Energy drop: {curr_loud:.0f} LUFS (from {prev_loud:.0f})",
                "from_state": f"{prev_loud:.0f} LUFS",
                "to_state": f"{curr_loud:.0f} LUFS",
                "raw_ratio": ratio,
            }

        return {"magnitude": magnitude, "type": TransitionType.UNKNOWN, "raw_ratio": ratio}

    def _compute_spectral_shift(self, prev: Dict, curr: Dict) -> Dict:
        """Detect timbral change via spectral centroid shift."""
        prev_cent = prev.get("spectral_centroid", 2000)
        curr_cent = curr.get("spectral_centroid", 2000)
        avg = (prev_cent + curr_cent) / 2 + 1e-8
        shift = abs(curr_cent - prev_cent) / max(avg, 1.0)

        return {
            "magnitude": min(1.0, shift * 2),
            "type": TransitionType.SPECTRAL_SHIFT,
            "detail": f"Spectral shift: {prev_cent:.0f} → {curr_cent:.0f} Hz centroid",
            "from_state": f"{prev_cent:.0f} Hz",
            "to_state": f"{curr_cent:.0f} Hz",
        }

    def _compute_harmonic_shift(self, prev: Dict, curr: Dict) -> Dict:
        """Detect shift between harmonic (tonal) and percussive content."""
        prev_hr = prev.get("harmonic_ratio", 0.5)
        curr_hr = curr.get("harmonic_ratio", 0.5)
        delta = abs(curr_hr - prev_hr)

        detail = ""
        if curr_hr > prev_hr + 0.2:
            detail = "Shift toward tonal/harmonic content"
        elif curr_hr < prev_hr - 0.2:
            detail = "Shift toward percussive content"

        return {
            "magnitude": min(1.0, delta * 3),
            "type": TransitionType.HARMONIC_SHIFT,
            "detail": detail or f"Harmonic ratio: {prev_hr:.2f} → {curr_hr:.2f}",
            "from_state": f"harmonic={prev_hr:.2f}",
            "to_state": f"harmonic={curr_hr:.2f}",
        }

    def _detect_event_onset(self, prev: Dict, curr: Dict) -> Optional[Dict]:
        """Detect sudden appearance of a specific acoustic event."""
        prev_events = {
            tag.get("label", ""): tag.get("confidence", 0)
            for tag in (prev.get("event_tags") or [])
        }
        curr_events = {
            tag.get("label", ""): tag.get("confidence", 0)
            for tag in (curr.get("event_tags") or [])
        }

        # Find events that appeared or strongly increased
        for event, conf in curr_events.items():
            prev_conf = prev_events.get(event, 0)
            delta = conf - prev_conf
            if delta > 0.3 and conf > 0.3:
                return {
                    "magnitude": min(1.0, delta * 1.5),
                    "type": TransitionType.EVENT_ONSET,
                    "detail": f"Event onset: {event} ({prev_conf:.2f} → {conf:.2f})",
                    "from_state": prev.get("dominant_source", "unknown"),
                    "to_state": event,
                }

        return None

    # ── Post-processing ──────────────────────────────────────────────────

    @staticmethod
    def _suppress_nearby(
        points: List[AcousticChangePoint],
        min_gap: float = 3.0,
    ) -> List[AcousticChangePoint]:
        """Non-maximum suppression: keep strongest change point within min_gap window."""
        if not points:
            return points

        points.sort(key=lambda p: p.timestamp)
        suppressed = [points[0]]

        for cp in points[1:]:
            if cp.timestamp - suppressed[-1].timestamp < min_gap:
                # Keep the stronger one
                if cp.magnitude > suppressed[-1].magnitude:
                    suppressed[-1] = cp
            else:
                suppressed.append(cp)

        return suppressed


change_point_detector = ChangePointDetector()
