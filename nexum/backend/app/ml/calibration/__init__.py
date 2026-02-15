"""
Nexum Confidence Calibration Service.

Converts raw model output scores into calibrated, trustworthy probabilities.

Architecture:
  - Per-model calibrators: each model gets its own calibration mapping
  - Three calibration methods: Platt Scaling, Isotonic Regression, Temperature Scaling
  - Stores both raw + calibrated scores for debugging/research
  - Color-banded confidence tiers for UI display

Calibration Flow:
  1. Collect (raw_score, ground_truth) pairs from user feedback + manual labels
  2. Fit calibrator per model (runs on daily cron or triggered manually)
  3. Apply calibrator at inference time → calibrated_probability
  4. UI displays calibrated probability with confidence band

Confidence Bands:
  - 0.00–0.25  →  Weak       (gray)
  - 0.25–0.50  →  Low        (orange)
  - 0.50–0.75  →  Moderate   (yellow)
  - 0.75–0.90  →  Strong     (green)
  - 0.90–1.00  →  Very Strong (bright green)
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ═══════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════


class CalibrationMethod(str, Enum):
    PLATT = "platt"           # Logistic regression — binary tasks
    ISOTONIC = "isotonic"     # Non-parametric — weird score curves
    TEMPERATURE = "temperature"  # Single scalar T — neural net softmax


class ConfidenceBand(str, Enum):
    WEAK = "weak"
    LOW = "low"
    MODERATE = "moderate"
    STRONG = "strong"
    VERY_STRONG = "very_strong"


BAND_THRESHOLDS = [
    (0.00, 0.25, ConfidenceBand.WEAK),
    (0.25, 0.50, ConfidenceBand.LOW),
    (0.50, 0.75, ConfidenceBand.MODERATE),
    (0.75, 0.90, ConfidenceBand.STRONG),
    (0.90, 1.01, ConfidenceBand.VERY_STRONG),
]

BAND_COLORS = {
    ConfidenceBand.WEAK: "#6b6b7b",
    ConfidenceBand.LOW: "#f08c3a",
    ConfidenceBand.MODERATE: "#f5a623",
    ConfidenceBand.STRONG: "#3ddc84",
    ConfidenceBand.VERY_STRONG: "#2ecc71",
}

BAND_LABELS = {
    ConfidenceBand.WEAK: "Weak",
    ConfidenceBand.LOW: "Low",
    ConfidenceBand.MODERATE: "Moderate",
    ConfidenceBand.STRONG: "Strong",
    ConfidenceBand.VERY_STRONG: "Very Strong",
}


@dataclass
class CalibratedScore:
    """Result of calibrating a raw model score."""
    raw_score: float
    calibrated_probability: float
    band: ConfidenceBand
    band_label: str
    band_color: str
    model_name: str
    calibration_version: str

    def to_dict(self) -> Dict:
        return {
            "raw_score": round(self.raw_score, 4),
            "calibrated_probability": round(self.calibrated_probability, 4),
            "band": self.band.value,
            "band_label": self.band_label,
            "band_color": self.band_color,
            "model_name": self.model_name,
            "calibration_version": self.calibration_version,
        }


@dataclass
class CalibrationData:
    """Training data for a calibrator."""
    raw_scores: List[float] = field(default_factory=list)
    ground_truths: List[int] = field(default_factory=list)  # 0 or 1


# ═══════════════════════════════════════════════════════════════════════════
# Calibrator Implementations
# ═══════════════════════════════════════════════════════════════════════════


class PlattCalibrator:
    """Platt Scaling — logistic regression on raw scores.

    Best for: binary classification tasks (e.g., "is this pitch-shifted?")
    Lightweight, well-understood, parametric.
    """

    def __init__(self):
        self.a: float = 1.0  # slope
        self.b: float = 0.0  # intercept
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        """Fit Platt calibration parameters via MLE."""
        from scipy.optimize import minimize

        def neg_log_likelihood(params):
            a, b = params
            p = 1.0 / (1.0 + np.exp(-(a * raw_scores + b)))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

        result = minimize(neg_log_likelihood, [1.0, 0.0], method="Nelder-Mead")
        self.a, self.b = result.x
        self.fitted = True
        logger.info(f"Platt calibrator fitted: a={self.a:.4f}, b={self.b:.4f}")

    def calibrate(self, raw_score: float) -> float:
        if not self.fitted:
            return raw_score
        return float(1.0 / (1.0 + np.exp(-(self.a * raw_score + self.b))))


class IsotonicCalibrator:
    """Isotonic Regression — non-parametric monotonic mapping.

    Best for: scores with non-linear, non-sigmoidal calibration curves.
    More flexible than Platt, but needs more data (50+ samples).
    """

    def __init__(self):
        self._regressor = None
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        from sklearn.isotonic import IsotonicRegression

        self._regressor = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        )
        self._regressor.fit(raw_scores, labels)
        self.fitted = True
        logger.info("Isotonic calibrator fitted.")

    def calibrate(self, raw_score: float) -> float:
        if not self.fitted or self._regressor is None:
            return raw_score
        result = self._regressor.predict([raw_score])
        return float(np.clip(result[0], 0.0, 1.0))


class TemperatureCalibrator:
    """Temperature Scaling — single scalar T applied to logits.

    Best for: neural network softmax outputs.
    Extremely lightweight, preserves relative ordering.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.fitted = False

    def fit(self, raw_scores: np.ndarray, labels: np.ndarray):
        """Find optimal temperature T that minimizes NLL."""
        from scipy.optimize import minimize_scalar

        def nll(T):
            T = max(T, 0.01)
            scaled = raw_scores / max(T, 1e-8)
            p = 1.0 / (1.0 + np.exp(-scaled))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p))

        result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
        self.temperature = max(result.x, 0.01)
        self.fitted = True
        logger.info(f"Temperature calibrator fitted: T={self.temperature:.4f}")

    def calibrate(self, raw_score: float) -> float:
        if not self.fitted:
            return raw_score
        scaled = raw_score / max(self.temperature, 1e-8)
        return float(1.0 / (1.0 + np.exp(-scaled)))


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Service
# ═══════════════════════════════════════════════════════════════════════════

# Registry of known model outputs that need calibration
CALIBRATION_TARGETS = {
    # Audio Intelligence — Head B (Manipulation Detection)
    "audio.speed_anomaly": CalibrationMethod.PLATT,
    "audio.pitch_shift": CalibrationMethod.PLATT,
    "audio.compression": CalibrationMethod.ISOTONIC,
    "audio.reverb_echo": CalibrationMethod.PLATT,
    "audio.robotic_autotune": CalibrationMethod.PLATT,
    "audio.time_stretch": CalibrationMethod.PLATT,
    "audio.overall_manipulation": CalibrationMethod.ISOTONIC,
    # Audio Intelligence — Head A (Source Classification)
    "audio.music_probability": CalibrationMethod.TEMPERATURE,
    "audio.speech_probability": CalibrationMethod.TEMPERATURE,
    # OCR
    "ocr.confidence": CalibrationMethod.ISOTONIC,
    # Whisper
    "whisper.segment_confidence": CalibrationMethod.TEMPERATURE,
    # CLIP Vision
    "clip.label_confidence": CalibrationMethod.TEMPERATURE,
    # Search fusion
    "search.final_score": CalibrationMethod.ISOTONIC,
}


class CalibrationService:
    """
    Manages calibration models for all scoreable outputs.

    Usage:
        calibrated = calibration_service.calibrate("audio.pitch_shift", 0.81)
        # CalibratedScore(raw=0.81, calibrated=0.62, band="moderate", ...)

    Training:
        calibration_service.add_sample("audio.pitch_shift", raw=0.81, truth=1)
        calibration_service.fit("audio.pitch_shift")
        calibration_service.save()
    """

    def __init__(self):
        self._calibrators: Dict[str, object] = {}
        self._training_data: Dict[str, CalibrationData] = {}
        self._version: str = "v0-uncalibrated"
        self._calibrators_dir = Path(settings.data_dir) / "calibrators"
        self._loaded = False

    @property
    def version(self) -> str:
        return self._version

    def _ensure_loaded(self):
        """Lazy-load persisted calibrators from disk."""
        if self._loaded:
            return
        self._loaded = True
        self._calibrators_dir.mkdir(parents=True, exist_ok=True)

        version_file = self._calibrators_dir / "version.txt"
        if version_file.exists():
            self._version = version_file.read_text().strip()

        for target_name, method in CALIBRATION_TARGETS.items():
            cal_path = self._calibrators_dir / f"{target_name.replace('.', '_')}.pkl"
            if cal_path.exists():
                try:
                    with open(cal_path, "rb") as f:
                        self._calibrators[target_name] = pickle.load(f)
                    logger.debug(f"Loaded calibrator: {target_name}")
                except Exception as e:
                    logger.warning(f"Failed to load calibrator {target_name}: {e}")

    def calibrate(self, target_name: str, raw_score: float) -> CalibratedScore:
        """Calibrate a raw model score → CalibratedScore with band."""
        self._ensure_loaded()

        calibrated = raw_score  # fallback: pass-through

        calibrator = self._calibrators.get(target_name)
        if calibrator and hasattr(calibrator, "calibrate") and hasattr(calibrator, "fitted") and calibrator.fitted:
            try:
                calibrated = calibrator.calibrate(raw_score)
            except Exception as e:
                logger.warning(f"Calibration failed for {target_name}: {e}")
                calibrated = raw_score

        # Clamp to [0, 1]
        calibrated = max(0.0, min(1.0, calibrated))

        # Determine confidence band
        band = ConfidenceBand.WEAK
        for lo, hi, b in BAND_THRESHOLDS:
            if lo <= calibrated < hi:
                band = b
                break

        return CalibratedScore(
            raw_score=raw_score,
            calibrated_probability=calibrated,
            band=band,
            band_label=BAND_LABELS[band],
            band_color=BAND_COLORS[band],
            model_name=target_name,
            calibration_version=self._version,
        )

    def calibrate_dict(self, target_prefix: str, scores: Dict[str, float]) -> Dict[str, CalibratedScore]:
        """Calibrate all scores in a dictionary with shared prefix."""
        result = {}
        for key, raw in scores.items():
            full_key = f"{target_prefix}.{key}"
            if full_key in CALIBRATION_TARGETS:
                result[key] = self.calibrate(full_key, raw)
            else:
                # No calibrator registered — pass-through with band
                result[key] = self.calibrate(full_key, raw)
        return result

    # ── Training Interface ───────────────────────────────────────────────

    def add_sample(self, target_name: str, raw_score: float, ground_truth: int):
        """Add a training sample for calibration fitting."""
        data = self._training_data.setdefault(target_name, CalibrationData())
        data.raw_scores.append(raw_score)
        data.ground_truths.append(ground_truth)

    def fit(self, target_name: str, min_samples: int = 30) -> bool:
        """Fit calibrator for a specific target using collected training data."""
        data = self._training_data.get(target_name)
        if not data or len(data.raw_scores) < min_samples:
            logger.warning(
                f"Insufficient data for {target_name}: "
                f"{len(data.raw_scores) if data else 0} < {min_samples}"
            )
            return False

        method = CALIBRATION_TARGETS.get(target_name, CalibrationMethod.PLATT)
        raw = np.array(data.raw_scores)
        labels = np.array(data.ground_truths)

        try:
            if method == CalibrationMethod.PLATT:
                calibrator = PlattCalibrator()
            elif method == CalibrationMethod.ISOTONIC:
                calibrator = IsotonicCalibrator()
            elif method == CalibrationMethod.TEMPERATURE:
                calibrator = TemperatureCalibrator()
            else:
                calibrator = PlattCalibrator()

            calibrator.fit(raw, labels)
            self._calibrators[target_name] = calibrator
            logger.info(f"Fitted calibrator for {target_name} using {method.value} ({len(raw)} samples)")
            return True

        except Exception as e:
            logger.error(f"Failed to fit calibrator for {target_name}: {e}")
            return False

    def fit_all(self, min_samples: int = 30) -> Dict[str, bool]:
        """Fit all calibrators that have sufficient training data."""
        results = {}
        for target_name in CALIBRATION_TARGETS:
            results[target_name] = self.fit(target_name, min_samples)

        # Update version
        import hashlib
        import time
        self._version = f"v{int(time.time())}-{hashlib.md5(str(results).encode()).hexdigest()[:6]}"

        return results

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self):
        """Persist all fitted calibrators to disk."""
        self._calibrators_dir.mkdir(parents=True, exist_ok=True)

        for target_name, calibrator in self._calibrators.items():
            if hasattr(calibrator, "fitted") and calibrator.fitted:
                cal_path = self._calibrators_dir / f"{target_name.replace('.', '_')}.pkl"
                try:
                    with open(cal_path, "wb") as f:
                        pickle.dump(calibrator, f)
                except Exception as e:
                    logger.warning(f"Failed to save calibrator {target_name}: {e}")

        version_file = self._calibrators_dir / "version.txt"
        version_file.write_text(self._version)
        logger.info(f"Saved {len(self._calibrators)} calibrators (version: {self._version})")

    def load_training_data(self, path: str):
        """Load training data from JSON file."""
        try:
            with open(path) as f:
                data = json.load(f)
            for target_name, samples in data.items():
                for sample in samples:
                    self.add_sample(target_name, sample["raw"], sample["truth"])
            logger.info(f"Loaded training data from {path}")
        except Exception as e:
            logger.error(f"Failed to load training data: {e}")

    # ── Diagnostics ──────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return calibration status for all targets."""
        self._ensure_loaded()
        status = {}
        for target_name, method in CALIBRATION_TARGETS.items():
            cal = self._calibrators.get(target_name)
            fitted = bool(cal and hasattr(cal, "fitted") and cal.fitted)
            data = self._training_data.get(target_name)
            status[target_name] = {
                "method": method.value,
                "fitted": fitted,
                "training_samples": len(data.raw_scores) if data else 0,
            }
        return {
            "version": self._version,
            "targets": status,
        }


calibration_service = CalibrationService()
