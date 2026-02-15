"""
Nexum OCR Service v4 — multilingual dual-engine OCR with temporal smoothing.

Engine 1: EasyOCR  (neural, multi-language, 80+ languages)
Engine 2: Tesseract (traditional, multi-language, cross-validation)

v4 Upgrades:
  - Language group batching: loads language families sequentially to fit 12 GB VRAM
  - Auto language detection: pre-classifies script type before full OCR
  - All EasyOCR + Tesseract supported languages enabled
  - Consensus scoring + temporal smoothing unchanged from v3.1
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Language Groups (loaded one at a time to fit VRAM) ───────────────────

LANGUAGE_GROUPS: Dict[str, List[str]] = {
    "latin_core": ["en", "es", "fr", "de", "pt", "it", "nl", "pl", "cs", "ro", "hr", "sk", "sl", "hu", "et", "lt", "lv", "da", "no", "sv", "fi", "is"],
    "latin_extended": ["tr", "az", "uz", "id", "ms", "tl", "vi", "sw", "sq", "cy", "ga", "mt", "eu", "gl", "ca", "la", "eo"],
    "arabic_persian": ["ar", "fa", "ur"],
    "cjk": ["ch_sim", "ch_tra", "ja", "ko"],
    "south_asian": ["hi", "bn", "ta", "te", "kn", "ml", "mr", "ne", "si"],
    "cyrillic": ["ru", "uk", "be", "bg", "sr", "mk", "mn"],
    "other_scripts": ["th", "ka", "el", "he"],
}


@dataclass
class OCRResult:
    text: str
    confidence: float
    bounding_box: Optional[List[Tuple[int, int]]] = None
    engine: str = ""
    language: str = ""


@dataclass
class MergedOCRResult:
    text: str
    confidence: float
    numeric_tokens: List[str] = field(default_factory=list)
    raw_results: List[OCRResult] = field(default_factory=list)
    detected_languages: List[str] = field(default_factory=list)


class OCRService:
    """Multilingual dual-engine OCR with language group batching."""

    _easyocr_readers: Dict[str, object] = {}
    _active_group: Optional[str] = None

    def __init__(self):
        self.threshold = settings.ocr_confidence_threshold

    # ── Script Detection ─────────────────────────────────────────────────

    @staticmethod
    def detect_script_groups(image: np.ndarray) -> List[str]:
        """
        Detect which script families are likely present using pixel analysis.
        Returns language groups to load. Always includes latin_core as fallback.
        """
        groups = ["latin_core"]  # always try Latin

        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Simple heuristic: check if image has content that suggests non-Latin scripts
        # In production, a small classifier would be better; here we check character density
        # and edge patterns that correlate with different scripts
        h, w = gray.shape[:2]
        if h < 20 or w < 20:
            return groups

        # Count edge density in different regions (CJK tends to be denser)
        edges = cv2.Canny(gray, 50, 150)
        density = np.mean(edges > 0)

        # If significant text content detected, also try other common groups
        if density > 0.05:
            groups.append("arabic_persian")
        if density > 0.08:
            groups.append("cjk")

        return groups

    # ── Engine 1: EasyOCR (batched by language group) ────────────────────

    def _get_easyocr(self, group: str = "latin_core"):
        """Get or create EasyOCR reader for a language group, unloading previous."""
        if group not in LANGUAGE_GROUPS:
            group = "latin_core"

        if group in self._easyocr_readers:
            return self._easyocr_readers[group]

        # Unload previous group to free VRAM
        if self._active_group and self._active_group != group:
            old = self._easyocr_readers.pop(self._active_group, None)
            if old:
                del old
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                logger.info(f"Unloaded EasyOCR group: {self._active_group}")

        import easyocr
        langs = LANGUAGE_GROUPS[group]
        gpu = settings.device == "cuda"
        logger.info(f"Loading EasyOCR group '{group}': {len(langs)} languages, gpu={gpu}")
        reader = easyocr.Reader(langs, gpu=gpu, verbose=False)
        self._easyocr_readers[group] = reader
        self._active_group = group
        return reader

    def run_easyocr(self, image: np.ndarray, group: str = "latin_core") -> List[OCRResult]:
        reader = self._get_easyocr(group)
        try:
            results = reader.readtext(image)
        except Exception as e:
            logger.warning(f"EasyOCR failed for group {group}: {e}")
            return []

        ocr_results = []
        for bbox, text, conf in results:
            if conf >= self.threshold and len(text.strip()) > 0:
                ocr_results.append(OCRResult(
                    text=text.strip(),
                    confidence=float(conf),
                    bounding_box=bbox,
                    engine="easyocr",
                    language=group,
                ))
        return ocr_results

    # ── Engine 2: Tesseract ──────────────────────────────────────────────

    def run_tesseract(self, image: np.ndarray) -> List[OCRResult]:
        try:
            import pytesseract

            lang_str = settings.tesseract_languages
            data = pytesseract.image_to_data(
                Image.fromarray(image),
                lang=lang_str,
                output_type=pytesseract.Output.DICT,
            )
            results = []
            for i, text in enumerate(data["text"]):
                conf = int(data["conf"][i]) / 100.0
                if conf >= self.threshold and len(text.strip()) > 1:
                    results.append(OCRResult(
                        text=text.strip(),
                        confidence=conf,
                        engine="tesseract",
                    ))
            return results
        except Exception as e:
            logger.warning(f"Tesseract failed: {e}")
            return []

    # ── Preprocessing ────────────────────────────────────────────────────

    @staticmethod
    def preprocess_frame(image: np.ndarray) -> np.ndarray:
        """Contrast normalization + denoising for OCR."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return denoised

    @staticmethod
    def preprocess_frame_color(image: np.ndarray) -> np.ndarray:
        """Color preprocessing for EasyOCR (works better with color input)."""
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened

    # ── Dual Engine Merge ────────────────────────────────────────────────

    def extract_ocr(self, image: np.ndarray, language_hint: Optional[str] = None) -> MergedOCRResult:
        """
        Run both engines across detected script groups and merge with consensus.

        v4: automatically detects script families present in the image and loads
        the appropriate language group models sequentially.
        """
        preprocessed_gray = self.preprocess_frame(image)
        preprocessed_color = self.preprocess_frame_color(image)

        # Determine which language groups to try
        if language_hint and language_hint in LANGUAGE_GROUPS:
            groups = [language_hint]
        else:
            groups = self.detect_script_groups(image)

        # Run EasyOCR across relevant language groups
        easy_results: List[OCRResult] = []
        for group in groups:
            group_results = self.run_easyocr(preprocessed_color, group)
            easy_results.extend(group_results)
            # If first group found good results, skip others (performance)
            if len(group_results) >= 3 and any(r.confidence > 0.7 for r in group_results):
                break

        # Run Tesseract (all configured languages at once — Tesseract handles this natively)
        tess_results = self.run_tesseract(preprocessed_gray)

        all_results = easy_results + tess_results

        if not all_results:
            return MergedOCRResult(text="", confidence=0.0)

        # Merge: boost confidence if both engines agree on a token
        easy_tokens = {r.text.lower() for r in easy_results}
        tess_tokens = {r.text.lower() for r in tess_results}
        consensus = easy_tokens & tess_tokens

        merged_texts = []
        total_conf = 0.0
        seen: Set[str] = set()
        detected_langs: Set[str] = set()

        for r in sorted(all_results, key=lambda x: -x.confidence):
            tok_lower = r.text.lower()
            if tok_lower in seen:
                continue
            seen.add(tok_lower)

            # Consensus boost: 20% if both engines agree
            boost = 1.20 if tok_lower in consensus else 1.0

            # Numeric boost: numbers are high-value OCR targets
            if re.match(r"^\d[\d,.:]*\d?$", r.text):
                boost *= 1.10

            adj_conf = min(r.confidence * boost, 1.0)
            merged_texts.append(r.text)
            total_conf += adj_conf
            if r.language:
                detected_langs.add(r.language)

        merged_text = " ".join(merged_texts)
        avg_conf = total_conf / len(seen) if seen else 0.0

        # Extract numeric tokens (scores, stats, timestamps)
        numeric_tokens = re.findall(r"\d[\d,.:]*\d|\d+", merged_text)

        return MergedOCRResult(
            text=merged_text,
            confidence=avg_conf,
            numeric_tokens=numeric_tokens,
            raw_results=all_results,
            detected_languages=sorted(detected_langs),
        )

    # ── Temporal Smoothing ───────────────────────────────────────────────

    @staticmethod
    def temporal_smooth(
        results: List[MergedOCRResult], window: int = 3
    ) -> List[MergedOCRResult]:
        """Boost tokens that persist across adjacent frames. Suppress transient noise."""
        if len(results) < 2:
            return results

        smoothed = []
        for i, r in enumerate(results):
            if not r.text.strip():
                smoothed.append(r)
                continue

            start = max(0, i - window // 2)
            end = min(len(results), i + window // 2 + 1)
            neighbor_tokens: Set[str] = set()
            for j in range(start, end):
                if j != i:
                    neighbor_tokens.update(results[j].text.lower().split())

            current_tokens = r.text.split()
            persistent_tokens = [
                t for t in current_tokens
                if t.lower() in neighbor_tokens or re.match(r"\d+", t)
            ]

            if persistent_tokens:
                smoothed.append(MergedOCRResult(
                    text=" ".join(persistent_tokens),
                    confidence=min(r.confidence * 1.10, 1.0),
                    numeric_tokens=r.numeric_tokens,
                    raw_results=r.raw_results,
                    detected_languages=r.detected_languages,
                ))
            else:
                smoothed.append(MergedOCRResult(
                    text=r.text,
                    confidence=r.confidence * 0.75,
                    numeric_tokens=r.numeric_tokens,
                    raw_results=r.raw_results,
                    detected_languages=r.detected_languages,
                ))

        return smoothed

    def unload(self):
        """Release all loaded models."""
        for group, reader in self._easyocr_readers.items():
            del reader
        self._easyocr_readers.clear()
        self._active_group = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("OCR service: all models unloaded")


ocr_service = OCRService()
