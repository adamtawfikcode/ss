"""
Nexum ML Embedding Service — generates text and visual embeddings.

Text:   intfloat/multilingual-e5-large  (1024-dim, 100+ languages)
Visual: OpenCLIP ViT-H-14               (1024-dim)

GPU memory strategy:
  When sequential_gpu is True, the CLIP model is loaded on demand and can be
  released via release_clip() to free VRAM for Whisper.  The text model is
  small enough to stay resident (or runs on CPU).
"""
from __future__ import annotations

import hashlib
import logging
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Singleton embedding service with lazy model loading and VRAM management."""

    _instance: Optional["EmbeddingService"] = None
    _text_model = None
    _clip_model = None
    _clip_preprocess = None
    _clip_tokenizer = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def _device(self) -> str:
        return settings.device

    # ── Text Embeddings ──────────────────────────────────────────────────

    def _load_text_model(self):
        from sentence_transformers import SentenceTransformer

        model_name = settings.text_embedding_model
        # Text model is small — keep on GPU if available, or CPU
        device = self._device
        logger.info(f"Loading text embedding model: {model_name} on {device}")
        self._text_model = SentenceTransformer(model_name, device=device)
        logger.info("Text embedding model loaded.")

    def embed_texts(self, texts: List[str], normalize: bool = True, is_query: bool = False) -> np.ndarray:
        """Embed a batch of text strings → (N, D) float16 array.

        For e5 models, queries need 'query: ' prefix and passages need 'passage: '.
        """
        if self._text_model is None:
            self._load_text_model()

        # e5 models require prefixed text for optimal performance
        model_name = settings.text_embedding_model.lower()
        if "e5" in model_name:
            prefix = "query: " if is_query else "passage: "
            texts = [f"{prefix}{t}" for t in texts]

        embeddings = self._text_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float16)

    def embed_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """Single text → (D,) float16 vector."""
        return self.embed_texts([text], is_query=is_query)[0]

    # ── Visual Embeddings (CLIP) ─────────────────────────────────────────

    def _load_clip_model(self):
        import open_clip

        model_name = settings.clip_model
        pretrained = settings.clip_pretrained
        device = self._device
        logger.info(f"Loading CLIP model: {model_name} ({pretrained}) on {device}")
        self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )
        self._clip_tokenizer = open_clip.get_tokenizer(model_name)
        self._clip_model.eval()
        logger.info("CLIP model loaded.")

    @property
    def clip_loaded(self) -> bool:
        return self._clip_model is not None

    def ensure_clip(self):
        """Explicitly load CLIP (called before vision / embedding passes)."""
        if self._clip_model is None:
            self._load_clip_model()

    def release_clip(self):
        """Free CLIP from GPU to make room for other models (e.g. Whisper)."""
        if self._clip_model is not None:
            del self._clip_model
            del self._clip_preprocess
            del self._clip_tokenizer
            self._clip_model = None
            self._clip_preprocess = None
            self._clip_tokenizer = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("CLIP model released from memory.")

    def release_text_model(self):
        """Free text model from memory."""
        if self._text_model is not None:
            del self._text_model
            self._text_model = None
            if self._device == "cuda":
                torch.cuda.empty_cache()
            logger.info("Text embedding model released from memory.")

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image], normalize: bool = True) -> np.ndarray:
        """Embed PIL images → (N, D) float16 array."""
        self.ensure_clip()

        device = self._device
        batch_size = 8  # reduced for ViT-H-14 (~8 GB VRAM)
        all_features = []

        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            preprocessed = torch.stack([self._clip_preprocess(img) for img in batch]).to(device)
            features = self._clip_model.encode_image(preprocessed)
            if normalize:
                features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())

        combined = torch.cat(all_features, dim=0)
        return combined.numpy().astype(np.float16)

    @torch.no_grad()
    def embed_image(self, image: Image.Image) -> np.ndarray:
        return self.embed_images([image])[0]

    @torch.no_grad()
    def embed_text_clip(self, text: str) -> np.ndarray:
        """Embed text using CLIP text encoder (for cross-modal search)."""
        self.ensure_clip()

        tokens = self._clip_tokenizer([text]).to(self._device)
        features = self._clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy().astype(np.float16)[0]

    # ── CLIP accessors for VisionService ─────────────────────────────────

    @property
    def clip_model(self):
        self.ensure_clip()
        return self._clip_model

    @property
    def clip_preprocess(self):
        self.ensure_clip()
        return self._clip_preprocess

    @property
    def clip_tokenizer(self):
        self.ensure_clip()
        return self._clip_tokenizer

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def vector_hash(vec: np.ndarray) -> str:
        """Deterministic hash for deduplication."""
        return hashlib.sha256(vec.tobytes()).hexdigest()[:16]

    @property
    def text_dim(self) -> int:
        return settings.text_embedding_dim

    @property
    def visual_dim(self) -> int:
        return settings.visual_embedding_dim

    @property
    def current_model_version(self) -> str:
        return f"{settings.text_embedding_model}|{settings.clip_model}-{settings.clip_pretrained}"


# Module-level singleton
embedding_service = EmbeddingService()
