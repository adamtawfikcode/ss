"""
Nexum Vision Service — visual understanding via CLIP zero-shot classification.

Shares the CLIP model from EmbeddingService (no double-loading).
Uses ViT-H-14 for highest accuracy zero-shot classification.

Features:
  - 40+ object labels, 25+ scene labels, 20+ activity/style labels
  - Expanded label vocabulary for deeper tagging
  - Blur detection with configurable threshold
  - Histogram-based scene change detection
  - Batch classification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Label Sets (expanded for maximum detail) ─────────────────────────────

OBJECT_LABELS = [
    "person", "people", "crowd", "face", "hand", "baby", "child",
    "dog", "cat", "bird", "horse", "fish",
    "car", "bicycle", "motorcycle", "airplane", "boat", "train", "bus", "truck",
    "food", "pizza", "burger", "sushi", "cake", "coffee", "drink", "bottle",
    "computer", "laptop", "phone", "tablet", "monitor", "screen", "keyboard", "mouse",
    "book", "newspaper", "document", "whiteboard", "chalkboard",
    "desk", "chair", "table", "bed", "couch", "shelf",
    "microphone", "camera", "headphones", "speaker",
    "guitar", "piano", "drums", "violin",
    "ball", "trophy", "medal", "flag",
    "building", "house", "bridge", "tower", "skyscraper",
    "tree", "flower", "mountain", "ocean", "river", "sky", "cloud", "sun", "moon",
    "game controller", "video game screen", "text overlay", "logo", "subtitle",
    "chart", "graph", "diagram", "map", "code on screen",
    "weapon", "money", "clock", "mirror", "painting",
]

SCENE_LABELS = [
    "indoor office", "outdoor nature", "classroom", "lecture hall",
    "kitchen", "bedroom", "living room", "bathroom",
    "studio", "recording studio", "podcast studio",
    "stage", "concert venue", "theater",
    "street", "sidewalk", "highway", "parking lot",
    "gym", "sports field", "swimming pool", "stadium",
    "restaurant", "cafe", "bar",
    "conference room", "meeting room", "coworking space",
    "gaming setup", "streaming setup",
    "laboratory", "hospital", "clinic",
    "warehouse", "factory", "garage", "workshop",
    "park", "garden", "beach", "forest", "desert",
    "rooftop", "balcony", "library", "museum",
    "airport", "train station", "subway",
    "store", "mall", "market",
]

ACTIVITY_LABELS = [
    "talking", "presenting", "lecturing", "explaining",
    "coding", "typing", "programming",
    "playing video games", "streaming",
    "cooking", "baking", "eating", "drinking",
    "exercising", "yoga", "running", "weight lifting", "stretching",
    "dancing", "singing", "rapping", "playing instrument",
    "drawing", "painting", "sculpting", "crafting",
    "writing on whiteboard", "writing on paper",
    "interviewing", "debating", "arguing", "discussing",
    "walking", "running", "jumping", "climbing", "swimming",
    "driving", "cycling", "skateboarding",
    "reading", "studying", "researching",
    "watching screen", "scrolling phone",
    "unboxing", "reviewing product", "testing",
    "building", "assembling", "repairing", "soldering",
    "filming", "photographing",
    "meditating", "praying", "sleeping",
]

STYLE_LABELS = [
    "vlog style", "professional production", "cinematic",
    "tutorial", "how-to", "educational",
    "podcast", "radio show",
    "news broadcast", "documentary", "investigative",
    "animation", "cartoon", "anime",
    "screencast", "screen recording", "software demo",
    "surveillance footage", "dashcam", "bodycam",
    "amateur recording", "phone recording", "selfie video",
    "music video", "lyric video",
    "advertisement", "promotional",
    "reaction video", "commentary",
    "timelapse", "slow motion",
]


@dataclass
class VisualTag:
    label: str
    confidence: float
    category: str  # "object", "scene", "activity", "style"


@dataclass
class FrameAnalysis:
    tags: List[VisualTag]
    scene_label: str
    is_blurry: bool
    blur_score: float
    top_objects: List[str]


class VisionService:
    """CLIP-based zero-shot visual understanding.

    Shares the CLIP model with EmbeddingService to avoid loading it twice.
    """

    _text_features_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}
    _labels_encoded: bool = False

    def _ensure_labels(self):
        """Encode all label sets using the shared CLIP model."""
        if self._labels_encoded:
            return

        from app.ml.embeddings.embedding_service import embedding_service
        embedding_service.ensure_clip()

        self._encode_labels("object", OBJECT_LABELS, embedding_service)
        self._encode_labels("scene", SCENE_LABELS, embedding_service)
        self._encode_labels("activity", ACTIVITY_LABELS, embedding_service)
        self._encode_labels("style", STYLE_LABELS, embedding_service)
        self._labels_encoded = True
        logger.info("Vision label embeddings cached.")

    @torch.no_grad()
    def _encode_labels(self, category: str, labels: List[str], emb_svc):
        prompts = [f"a photo of {label}" for label in labels]
        device = settings.device
        tokens = emb_svc.clip_tokenizer(prompts).to(device)
        features = emb_svc.clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        self._text_features_cache[category] = (labels, features)

    def reset_cache(self):
        """Call after CLIP is released/reloaded to re-encode labels."""
        self._text_features_cache.clear()
        self._labels_encoded = False

    @torch.no_grad()
    def classify_frame(
        self,
        image: Image.Image,
        top_k: int = 7,
        min_confidence: float = 0.06,
    ) -> FrameAnalysis:
        """Classify a single frame across all label sets."""
        from app.ml.embeddings.embedding_service import embedding_service

        self._ensure_labels()

        # Check blur first
        img_np = np.array(image)
        blur_score = self._compute_blur(img_np)
        is_blurry = blur_score < settings.blur_threshold

        # Encode image using shared CLIP
        device = settings.device
        img_tensor = embedding_service.clip_preprocess(image).unsqueeze(0).to(device)
        img_features = embedding_service.clip_model.encode_image(img_tensor)
        img_features = img_features / img_features.norm(dim=-1, keepdim=True)

        all_tags: List[VisualTag] = []

        for category, (labels, text_features) in self._text_features_cache.items():
            similarities = (img_features @ text_features.T).squeeze(0)
            probs = similarities.softmax(dim=-1).cpu().numpy()

            top_indices = probs.argsort()[-top_k:][::-1]
            for idx in top_indices:
                conf = float(probs[idx])
                if conf >= min_confidence:
                    all_tags.append(VisualTag(
                        label=labels[idx],
                        confidence=round(conf, 4),
                        category=category,
                    ))

        # Primary scene
        scene_tags = [t for t in all_tags if t.category == "scene"]
        scene_label = scene_tags[0].label if scene_tags else "unknown"

        # Top objects
        object_tags = sorted(
            [t for t in all_tags if t.category == "object"],
            key=lambda x: -x.confidence,
        )
        top_objects = [t.label for t in object_tags[:7]]

        return FrameAnalysis(
            tags=sorted(all_tags, key=lambda x: -x.confidence),
            scene_label=scene_label,
            is_blurry=is_blurry,
            blur_score=blur_score,
            top_objects=top_objects,
        )

    def classify_frames_batch(
        self, images: List[Image.Image], top_k: int = 7
    ) -> List[FrameAnalysis]:
        """Batch classification — processes one at a time to limit VRAM."""
        return [self.classify_frame(img, top_k) for img in images]

    @staticmethod
    def _compute_blur(image_np: np.ndarray) -> float:
        """Laplacian variance — higher = sharper."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY) if len(image_np.shape) == 3 else image_np
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def detect_scene_change(
        frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 30.0
    ) -> bool:
        """Histogram-based scene change detection."""
        hist_a = cv2.calcHist([frame_a], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist_b = cv2.calcHist([frame_b], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        diff = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CHISQR)
        return diff > threshold


vision_service = VisionService()
