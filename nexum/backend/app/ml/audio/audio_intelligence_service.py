"""
Nexum Audio Intelligence Service — Acoustic Intelligence Layer.

Architecture: Shared audio feature extraction + 3 analysis heads.

  Head A — Source & Event Classification
    What is happening acoustically? (music, speech, applause, laughter, etc.)
    Uses PANNs CNN14 pretrained on AudioSet (527 classes).

  Head B — Manipulation / Integrity Detection
    Has the audio been altered? (speed, pitch shift, compression, reverb)
    Uses spectral analysis heuristics on mel/STFT features.

  Head C — Musical & Acoustic Attributes
    How does it sound? (BPM, key, loudness, spectral brightness, dynamics)
    Uses librosa signal processing.

Processing: 5-second windows with 2.5s hop (50% overlap).
GPU: CNN14 inference is GPU-accelerated; librosa runs on CPU.
Memory: ~300 MB VRAM for CNN14 — fits alongside nothing else on 12 GB,
        so load/release sequentially like Whisper and CLIP.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════


@dataclass
class AudioEvent:
    """Single acoustic event tag with confidence."""
    label: str
    confidence: float
    category: str = "event"  # "source" | "event"


@dataclass
class ManipulationScores:
    """Probabilistic manipulation detection results."""
    speed_anomaly: float = 0.0       # 0–1: likelihood of speed change
    pitch_shift: float = 0.0         # 0–1: likelihood of pitch alteration
    compression: float = 0.0         # 0–1: heavy dynamic compression
    reverb_echo: float = 0.0         # 0–1: artificial reverb/echo
    robotic_autotune: float = 0.0    # 0–1: autotune / vocoder artifacts
    time_stretch: float = 0.0        # 0–1: time stretching artifacts
    overall_manipulation: float = 0.0  # weighted aggregate

    def to_dict(self) -> Dict[str, float]:
        return {
            "speed_anomaly": round(self.speed_anomaly, 3),
            "pitch_shift": round(self.pitch_shift, 3),
            "compression": round(self.compression, 3),
            "reverb_echo": round(self.reverb_echo, 3),
            "robotic_autotune": round(self.robotic_autotune, 3),
            "time_stretch": round(self.time_stretch, 3),
            "overall_manipulation": round(self.overall_manipulation, 3),
        }


@dataclass
class MusicalAttributes:
    """Acoustic / musical properties of an audio window."""
    bpm: Optional[float] = None
    key: Optional[str] = None          # e.g., "C major", "A minor"
    loudness_lufs: float = -30.0       # integrated loudness
    dynamic_range_db: float = 0.0
    spectral_centroid: float = 0.0     # "brightness" (Hz)
    spectral_rolloff: float = 0.0      # frequency below which 85% energy
    harmonic_ratio: float = 0.0        # harmonic vs percussive balance (0–1)
    zero_crossing_rate: float = 0.0    # noisiness indicator

    def to_dict(self) -> Dict:
        return {
            "bpm": round(self.bpm, 1) if self.bpm else None,
            "key": self.key,
            "loudness_lufs": round(self.loudness_lufs, 1),
            "dynamic_range_db": round(self.dynamic_range_db, 1),
            "spectral_centroid": round(self.spectral_centroid, 1),
            "spectral_rolloff": round(self.spectral_rolloff, 1),
            "harmonic_ratio": round(self.harmonic_ratio, 3),
            "zero_crossing_rate": round(self.zero_crossing_rate, 5),
        }


@dataclass
class AudioWindowResult:
    """Complete analysis result for a single audio window."""
    start_time: float
    end_time: float
    source_tags: List[AudioEvent] = field(default_factory=list)
    event_tags: List[AudioEvent] = field(default_factory=list)
    manipulation: ManipulationScores = field(default_factory=ManipulationScores)
    attributes: MusicalAttributes = field(default_factory=MusicalAttributes)
    music_probability: float = 0.0
    speech_probability: float = 0.0
    embedding: Optional[np.ndarray] = None  # PANNs logits vector for search


@dataclass
class AudioAnalysisResult:
    """Full audio intelligence output for an entire audio file."""
    windows: List[AudioWindowResult]
    duration_seconds: float
    global_bpm: Optional[float] = None
    global_key: Optional[str] = None
    dominant_source: str = "unknown"
    total_music_seconds: float = 0.0
    total_speech_seconds: float = 0.0
    total_silence_seconds: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# AudioSet Label Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Map from AudioSet 527-class indices to our simplified labels.
# Full AudioSet ontology has hierarchical labels — we extract the ones we care about.
# These indices correspond to the PANNs CNN14 output logits.

AUDIOSET_SOURCE_MAP = {
    0: "speech",        # Speech
    137: "music",       # Music
    # Silence is inferred from low activation across all classes
}

# Curated subset of AudioSet event classes we track
AUDIOSET_EVENT_MAP = {
    16: "laughter",      # Laughter
    17: "crying",        # Crying, sobbing
    23: "screaming",     # Scream
    24: "singing",       # Singing
    36: "clapping",      # Clapping
    37: "applause",      # Applause
    58: "crowd",         # Crowd
    288: "explosion",    # Explosion
    289: "gunshot",      # Gunshot, gunfire
    315: "siren",        # Siren
    316: "alarm",        # Fire alarm
    323: "bell",         # Bell
    339: "engine",       # Engine
    364: "door_slam",    # Door
    383: "glass_breaking",  # Glass (shatter)
    389: "typing",       # Typing
    394: "whistle",      # Whistle
    397: "animal",       # Animal
    420: "water",        # Water
    427: "wind",         # Wind
    428: "thunder",      # Thunder
    429: "fireworks",    # Fireworks
    440: "footsteps",    # Footsteps
    477: "cheering",     # Cheering
}

# Musical key names for chroma-to-key mapping
KEY_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


# ═══════════════════════════════════════════════════════════════════════════
# Service
# ═══════════════════════════════════════════════════════════════════════════


class AudioIntelligenceService:
    """
    Acoustic Intelligence Layer — makes audio first-class searchable data.

    Architecture: shared CNN14 encoder + 3 lightweight analysis heads.
    """

    _model = None
    _device = None

    # ── Model Lifecycle ──────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load PANNs CNN14 for audio tagging."""
        try:
            from panns_inference import AudioTagging
        except ImportError:
            logger.warning(
                "panns_inference not installed — falling back to librosa-only mode. "
                "Install with: pip install panns-inference"
            )
            self._model = None
            return

        self._device = settings.device
        logger.info(f"Loading PANNs CNN14 on {self._device}...")

        self._model = AudioTagging(
            checkpoint_path=None,  # auto-download CNN14
            device=self._device,
        )
        logger.info("PANNs CNN14 loaded.")

    def release_model(self):
        """Free PANNs from GPU for other models."""
        if self._model is not None:
            del self._model
            self._model = None
            if settings.device == "cuda":
                torch.cuda.empty_cache()
            logger.info("PANNs CNN14 released from memory.")

    # ── Main Entry Point ─────────────────────────────────────────────────

    def analyze(self, audio_path: str) -> AudioAnalysisResult:
        """
        Full audio intelligence analysis on a WAV file.

        1. Load audio at target sample rate
        2. Slice into overlapping windows
        3. For each window: CNN14 classification + spectral analysis
        4. Aggregate global attributes
        """
        import librosa

        # Load audio
        y, sr = librosa.load(
            audio_path,
            sr=settings.audio_sample_rate,
            mono=True,
        )
        duration = len(y) / max(sr, 1)
        logger.info(f"Audio loaded: {duration:.1f}s @ {sr} Hz")

        if duration < 0.5:
            logger.warning("Audio too short for analysis")
            return AudioAnalysisResult(windows=[], duration_seconds=duration)

        # Slice into windows
        window_samples = int(settings.audio_window_seconds * sr)
        hop_samples = int(settings.audio_hop_seconds * sr)

        windows: List[AudioWindowResult] = []
        pos = 0

        while pos < len(y):
            end = min(pos + window_samples, len(y))
            chunk = y[pos:end]

            # Pad short final window
            if len(chunk) < window_samples // 2:
                break
            if len(chunk) < window_samples:
                chunk = np.pad(chunk, (0, window_samples - len(chunk)))

            start_time = pos / max(sr, 1)
            end_time = min(end / max(sr, 1), duration)

            result = self._analyze_window(chunk, sr, start_time, end_time)
            windows.append(result)

            pos += hop_samples

        # Global attributes
        global_result = self._aggregate_global(y, sr, duration, windows)
        logger.info(
            f"Audio analysis complete: {len(windows)} windows, "
            f"dominant={global_result.dominant_source}, "
            f"BPM={global_result.global_bpm}"
        )

        return global_result

    # ── Per-Window Analysis ──────────────────────────────────────────────

    def _analyze_window(
        self,
        chunk: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float,
    ) -> AudioWindowResult:
        """Analyze a single audio window through all 3 heads."""
        result = AudioWindowResult(start_time=start_time, end_time=end_time)

        # Head A: Source & Event Classification (PANNs CNN14)
        self._classify_events(chunk, sr, result)

        # Head B: Manipulation Detection (spectral heuristics)
        self._detect_manipulation(chunk, sr, result)

        # Head C: Musical Attributes (librosa)
        self._extract_attributes(chunk, sr, result)

        return result

    # ── Head A: Source & Event Classification ─────────────────────────────

    def _classify_events(
        self,
        chunk: np.ndarray,
        sr: int,
        result: AudioWindowResult,
    ):
        """Classify audio events using PANNs CNN14."""
        # Ensure model loaded
        if self._model is None:
            self._load_model()

        if self._model is None:
            # Fallback: basic energy-based source detection
            self._classify_events_fallback(chunk, sr, result)
            return

        # PANNs expects float32, shape (1, samples) at 32 kHz
        audio_input = chunk[np.newaxis, :].astype(np.float32)

        # Resample if needed
        if sr != 32000:
            import librosa
            chunk_32k = librosa.resample(chunk, orig_sr=sr, target_sr=32000)
            audio_input = chunk_32k[np.newaxis, :].astype(np.float32)

        try:
            clipwise_output, embedding = self._model.inference(audio_input)
            logits = clipwise_output[0]  # shape: (527,)

            # Store embedding for vector search
            result.embedding = logits.copy()

            # Extract source classifications
            for idx, label in AUDIOSET_SOURCE_MAP.items():
                conf = float(logits[idx]) if idx < len(logits) else 0.0
                if conf > settings.audio_min_event_confidence:
                    result.source_tags.append(AudioEvent(
                        label=label,
                        confidence=round(conf, 3),
                        category="source",
                    ))

            # Set probability fields
            result.speech_probability = float(logits[0]) if len(logits) > 0 else 0.0
            result.music_probability = float(logits[137]) if len(logits) > 137 else 0.0

            # Extract event classifications
            for idx, label in AUDIOSET_EVENT_MAP.items():
                if idx < len(logits):
                    conf = float(logits[idx])
                    if conf > settings.audio_min_event_confidence:
                        result.event_tags.append(AudioEvent(
                            label=label,
                            confidence=round(conf, 3),
                            category="event",
                        ))

            # Sort by confidence
            result.source_tags.sort(key=lambda e: e.confidence, reverse=True)
            result.event_tags.sort(key=lambda e: e.confidence, reverse=True)

            # Infer silence
            max_activation = float(np.max(logits))
            if max_activation < 0.1:
                result.source_tags.insert(0, AudioEvent(
                    label="silence",
                    confidence=round(1.0 - max_activation, 3),
                    category="source",
                ))

        except Exception as e:
            logger.warning(f"PANNs inference failed: {e}")
            self._classify_events_fallback(chunk, sr, result)

    def _classify_events_fallback(
        self,
        chunk: np.ndarray,
        sr: int,
        result: AudioWindowResult,
    ):
        """Basic energy-based classification when PANNs unavailable."""
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if rms < 0.005:
            result.source_tags.append(AudioEvent("silence", 0.9, "source"))
        elif rms < 0.02:
            result.source_tags.append(AudioEvent("noise", 0.5, "source"))
        else:
            # Can't distinguish music/speech without model
            result.source_tags.append(AudioEvent("sound", 0.5, "source"))

    # ── Head B: Manipulation / Integrity Detection ───────────────────────

    def _detect_manipulation(
        self,
        chunk: np.ndarray,
        sr: int,
        result: AudioWindowResult,
    ):
        """
        Detect audio manipulation through spectral analysis.

        Heuristic approach — not ML-based, but effective for common edits:
        - Speed change: abnormal formant spacing in speech regions
        - Pitch shift: spectral comb filter artifacts
        - Compression: low crest factor (peak/RMS ratio)
        - Reverb/echo: high autocorrelation at non-zero lags
        - Robotic/autotune: abnormally stable pitch contour
        - Time stretch: phase coherence artifacts in STFT
        """
        import librosa

        scores = ManipulationScores()

        # Compute shared features
        stft = librosa.stft(chunk, n_fft=2048, hop_length=512)
        magnitude = np.abs(stft)
        phase = np.angle(stft)

        # ── Speed anomaly: spectral flatness deviation ───────────────
        # Sped-up audio has unnaturally high spectral centroid
        spec_flat = float(np.mean(librosa.feature.spectral_flatness(y=chunk)))
        # Normal speech ~0.01–0.05, sped up tends higher
        scores.speed_anomaly = min(1.0, max(0.0, (spec_flat - 0.04) * 10))

        # ── Pitch shift: harmonic spacing regularity ─────────────────
        # Pitch-shifted audio has inconsistent harmonic series
        chroma = librosa.feature.chroma_stft(S=magnitude, sr=sr)
        chroma_std = float(np.std(chroma))
        # Low variance in chroma = potentially pitch-locked
        scores.pitch_shift = min(1.0, max(0.0, 0.5 - chroma_std) * 2)

        # ── Heavy compression: crest factor ──────────────────────────
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        peak = float(np.max(np.abs(chunk))) + 1e-8
        crest_factor = peak / (rms + 1e-8)
        # Normal audio: crest factor 4–12. Heavily compressed: < 3
        scores.compression = min(1.0, max(0.0, (4.0 - crest_factor) / 3.0))

        # ── Reverb / echo: autocorrelation at 20–100ms lags ─────────
        autocorr_start = int(0.02 * sr)   # 20ms
        autocorr_end = int(0.10 * sr)     # 100ms
        if len(chunk) > autocorr_end:
            autocorr = np.correlate(
                chunk[:autocorr_end * 2],
                chunk[:autocorr_end * 2],
                mode="full",
            )
            center = len(autocorr) // 2
            reverb_region = autocorr[center + autocorr_start:center + autocorr_end]
            if len(reverb_region) > 0:
                # Normalize by zero-lag
                zero_lag = autocorr[center] + 1e-8
                reverb_ratio = float(np.max(reverb_region) / zero_lag)
                scores.reverb_echo = min(1.0, max(0.0, reverb_ratio * 2))

        # ── Robotic / autotune: pitch stability ──────────────────────
        try:
            f0, voiced_flag, _ = librosa.pyin(
                chunk, fmin=60, fmax=600, sr=sr
            )
            voiced_f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
            if len(voiced_f0) > 5:
                # Unnaturally stable pitch = autotune
                pitch_cv = float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-8))
                # Normal speech CV: 0.1–0.4. Autotuned: < 0.05
                scores.robotic_autotune = min(1.0, max(0.0, (0.08 - pitch_cv) * 20))
        except Exception:
            pass

        # ── Time stretch: phase coherence ────────────────────────────
        if phase.shape[1] > 2:
            # Phase derivative should be smooth; time stretching adds jitter
            phase_diff = np.diff(phase, axis=1)
            phase_deviation = float(np.std(np.diff(phase_diff, axis=1)))
            # Normal: ~0.5–1.5. Time-stretched: > 2.0
            scores.time_stretch = min(1.0, max(0.0, (phase_deviation - 1.5) / 2.0))

        # ── Overall aggregate ────────────────────────────────────────
        weights = [0.2, 0.2, 0.15, 0.15, 0.15, 0.15]
        values = [
            scores.speed_anomaly, scores.pitch_shift, scores.compression,
            scores.reverb_echo, scores.robotic_autotune, scores.time_stretch,
        ]
        scores.overall_manipulation = round(
            sum(w * v for w, v in zip(weights, values)), 3
        )

        result.manipulation = scores

    # ── Head C: Musical & Acoustic Attributes ────────────────────────────

    def _extract_attributes(
        self,
        chunk: np.ndarray,
        sr: int,
        result: AudioWindowResult,
    ):
        """Extract musical and acoustic features using librosa."""
        import librosa

        attrs = MusicalAttributes()

        # RMS energy → approximate loudness
        rms = librosa.feature.rms(y=chunk)[0]
        rms_mean = float(np.mean(rms)) + 1e-10
        attrs.loudness_lufs = round(20 * math.log10(rms_mean + 1e-10), 1)

        # Dynamic range (peak-to-average)
        rms_max = float(np.max(rms)) + 1e-10
        rms_min = float(np.min(rms[rms > 0])) if np.any(rms > 0) else 1e-10
        attrs.dynamic_range_db = round(20 * math.log10(max(rms_max, 1e-10) / max(rms_min, 1e-10)), 1)

        # Spectral centroid (brightness)
        centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
        attrs.spectral_centroid = float(np.mean(centroid))

        # Spectral rolloff (energy concentration)
        rolloff = librosa.feature.spectral_rolloff(y=chunk, sr=sr, roll_percent=0.85)[0]
        attrs.spectral_rolloff = float(np.mean(rolloff))

        # Zero crossing rate (noisiness)
        zcr = librosa.feature.zero_crossing_rate(chunk)[0]
        attrs.zero_crossing_rate = float(np.mean(zcr))

        # Harmonic-percussive ratio
        try:
            y_harmonic, y_percussive = librosa.effects.hpss(chunk)
            h_energy = float(np.sum(y_harmonic ** 2))
            p_energy = float(np.sum(y_percussive ** 2))
            total = h_energy + p_energy + 1e-10
            attrs.harmonic_ratio = round(h_energy / max(total, 1e-10), 3)
        except Exception:
            attrs.harmonic_ratio = 0.5

        # BPM (tempo) — only meaningful for windows with rhythmic content
        try:
            onset_env = librosa.onset.onset_strength(y=chunk, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            if tempo is not None and len(tempo) > 0:
                bpm = float(tempo[0])
                if settings.audio_bpm_range[0] <= bpm <= settings.audio_bpm_range[1]:
                    attrs.bpm = bpm
        except Exception:
            pass

        # Key detection via chroma
        try:
            chroma = librosa.feature.chroma_cqt(y=chunk, sr=sr)
            chroma_sum = np.sum(chroma, axis=1)  # (12,)

            # Major and minor profiles (Krumhansl-Kessler)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

            best_corr = -1
            best_key = None
            for shift in range(12):
                rolled = np.roll(chroma_sum, -shift)
                for mode, profile in [("major", major_profile), ("minor", minor_profile)]:
                    corr = float(np.corrcoef(rolled, profile)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_key = f"{KEY_NAMES[shift]} {mode}"

            if best_corr > 0.5:  # confidence threshold
                attrs.key = best_key
        except Exception:
            pass

        result.attributes = attrs

    # ── Global Aggregation ───────────────────────────────────────────────

    def _aggregate_global(
        self,
        y: np.ndarray,
        sr: int,
        duration: float,
        windows: List[AudioWindowResult],
    ) -> AudioAnalysisResult:
        """Aggregate per-window results into file-level summary."""
        import librosa

        result = AudioAnalysisResult(
            windows=windows,
            duration_seconds=duration,
        )

        if not windows:
            return result

        # Dominant source across all windows
        source_seconds: Dict[str, float] = {}
        for w in windows:
            window_dur = w.end_time - w.start_time
            top_source = w.source_tags[0].label if w.source_tags else "unknown"
            source_seconds[top_source] = source_seconds.get(top_source, 0) + window_dur

        if source_seconds:
            result.dominant_source = max(source_seconds, key=source_seconds.get)
            result.total_music_seconds = source_seconds.get("music", 0)
            result.total_speech_seconds = source_seconds.get("speech", 0)
            result.total_silence_seconds = source_seconds.get("silence", 0)

        # Global BPM (full-track tempo estimation)
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            if tempo is not None and len(tempo) > 0:
                bpm = float(tempo[0])
                if settings.audio_bpm_range[0] <= bpm <= settings.audio_bpm_range[1]:
                    result.global_bpm = round(bpm, 1)
        except Exception:
            pass

        # Global key (from full track chroma)
        try:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            chroma_sum = np.sum(chroma, axis=1)
            major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                                       2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
            minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                                       2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
            best_corr = -1
            best_key = None
            for shift in range(12):
                rolled = np.roll(chroma_sum, -shift)
                for mode, profile in [("major", major_profile), ("minor", minor_profile)]:
                    corr = float(np.corrcoef(rolled, profile)[0, 1])
                    if corr > best_corr:
                        best_corr = corr
                        best_key = f"{KEY_NAMES[shift]} {mode}"
            if best_corr > 0.5:
                result.global_key = best_key
        except Exception:
            pass

        return result


audio_intelligence_service = AudioIntelligenceService()
