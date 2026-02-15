"""
Nexum Core Settings — v4.0: Calibrated Acoustic Intelligence Engine.

Upgrades from v3.1:
  - Confidence calibration across all model outputs
  - Audio-transcript alignment scoring
  - Acoustic change point detection
  - Model upgrades: ViT-H-14 (1024-dim), multilingual-e5-large (1024-dim)
  - 4K video + Opus audio downloads
  - All-language OCR (80+ languages)
  - 14-modality search fusion

Hardware target: NVIDIA RTX 5070 Ti (12 GB VRAM). Sequential GPU loading.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic_settings import BaseSettings, SettingsConfigDict


def _auto_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8",
        env_prefix="NEXUM_", case_sensitive=False,
    )

    # ── App ──────────────────────────────────────────────────────────────
    app_name: str = "Nexum"
    app_version: str = "4.1.0"
    debug: bool = False
    log_level: str = "INFO"
    secret_key: str = "change-me-in-production"
    api_prefix: str = "/api/v1"

    # ── PostgreSQL ───────────────────────────────────────────────────────
    db_host: str = "postgres"
    db_port: int = 5432
    db_user: str = "nexum"
    db_password: str = "nexum_secret"
    db_name: str = "nexum"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def database_url_sync(self) -> str:
        return (
            f"postgresql://{self.db_user}:{self.db_password}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    # ── Redis / Celery ───────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379/0"
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"

    # ── Qdrant ───────────────────────────────────────────────────────────
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    qdrant_collection_text: str = "nexum_text"
    qdrant_collection_visual: str = "nexum_visual"

    # v4: 1024-dim for both text and visual (upgraded models)
    text_embedding_dim: int = 1024
    visual_embedding_dim: int = 1024

    # ── MinIO / S3 ───────────────────────────────────────────────────────
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "nexum_minio"
    minio_secret_key: str = "nexum_minio_secret"
    minio_bucket: str = "nexum-data"
    minio_secure: bool = False

    # ── Crawler ──────────────────────────────────────────────────────────
    crawler_interval_seconds: int = 300
    crawler_max_concurrent: int = 4
    crawler_rate_limit_delay: float = 2.0
    channels_file: str = "/app/config/channels.txt"

    supported_platforms: List[str] = [
        "youtube", "tiktok", "twitter", "instagram",
        "twitch", "dailymotion", "vimeo", "reddit",
        "facebook", "bilibili",
    ]

    # ── Media Processing (Maximum Quality) ───────────────────────────────
    frame_sample_interval: float = 1.0
    segment_duration_seconds: int = 15
    segment_overlap_seconds: float = 2.0
    ocr_confidence_threshold: float = 0.25
    transcript_confidence_threshold: float = 0.3
    blur_threshold: float = 12.0
    max_video_duration_seconds: int = 14400
    min_video_duration_seconds: int = 3
    max_frames_per_video: int = 8000
    max_segments_per_video: int = 10000

    # Download: 4K video + Opus audio — maximum fidelity
    download_max_height: int = 2160
    download_format_youtube: str = (
        "bestvideo[height<=2160][ext=webm]+bestaudio[ext=webm]/"
        "bestvideo[height<=2160]+bestaudio/best[height<=2160]"
    )
    download_format_tiktok: str = "best"
    download_format_generic: str = "bestvideo[height<=2160]+bestaudio/best"
    download_prefer_opus: bool = True
    download_audio_format: str = "opus/vorbis/aac/best"

    # ── ML Models (Maximum Quality — all fit 12 GB sequentially) ─────────
    # Whisper large-v3: ~3 GB VRAM
    whisper_model: str = "large-v3"
    whisper_beam_size: int = 10
    whisper_best_of: int = 5
    whisper_compute_type: str = "float16"
    whisper_patience: float = 2.0
    whisper_length_penalty: float = 1.0
    whisper_suppress_blank: bool = True
    whisper_initial_prompt: Optional[str] = None

    # Text embeddings: multilingual-e5-large (1024-dim, 100+ languages)
    text_embedding_model: str = "intfloat/multilingual-e5-large"

    # CLIP: ViT-H-14 (1024-dim) — highest accuracy open CLIP model, ~8 GB
    clip_model: str = "ViT-H-14"
    clip_pretrained: str = "laion2b_s32b_b79k"

    # OCR: All languages supported by EasyOCR (80+)
    ocr_languages: List[str] = [
        "en", "fr", "de", "es", "it", "pt", "nl", "pl", "ro", "sv",
        "no", "da", "fi", "cs", "sk", "hr", "sl", "hu", "et", "lt", "lv",
        "tr", "az", "id", "ms", "vi", "tl",
        "ar", "fa", "ur",
        "ch_sim", "ch_tra", "ja", "ko",
        "hi", "bn", "ta", "te", "kn", "mr", "ne",
        "ru", "uk", "bg", "sr", "be", "mn",
        "th", "ka", "el", "he",
    ]
    tesseract_languages: str = (
        "eng+ara+fra+deu+spa+ita+por+nld+pol+rus+ukr+jpn+kor+"
        "chi_sim+chi_tra+hin+ben+tha+tur+vie+heb+ell"
    )

    device: str = _auto_device()
    sequential_gpu: bool = True

    # ── Search ───────────────────────────────────────────────────────────
    search_top_k: int = 200
    rerank_top_k: int = 30
    search_cache_ttl: int = 300

    # ── Fusion Weights (14 modalities, sum = 1.0) ────────────────────────
    weight_text_semantic: float = 0.20
    weight_visual_similarity: float = 0.12
    weight_ocr_match: float = 0.08
    weight_keyword_match: float = 0.07
    weight_temporal_coherence: float = 0.05
    weight_emotion_context: float = 0.02
    weight_comment_semantic: float = 0.08
    weight_comment_timestamp_boost: float = 0.03
    weight_agreement_cluster: float = 0.03
    weight_entity_overlap: float = 0.03
    weight_user_cluster_confidence: float = 0.03
    weight_audio_event_match: float = 0.08
    weight_audio_attribute_match: float = 0.07
    # v4 new modalities
    weight_alignment_quality: float = 0.06
    weight_change_point_proximity: float = 0.05

    # ── Neo4j ────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://neo4j:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "nexum_graph_secret"
    neo4j_database: str = "neo4j"
    neo4j_max_connection_pool_size: int = 50

    # ── Comment Ingestion ────────────────────────────────────────────────
    comment_fetch_top_n: int = 200
    comment_max_reply_depth: int = 20
    comment_min_likes_threshold: int = 0
    comment_min_reply_likes_threshold: int = 0
    comment_rate_limit_delay: float = 1.5
    comment_spam_threshold: float = 0.75
    comment_toxicity_threshold: float = 0.80
    comment_max_length_compress: int = 2000
    comment_batch_size: int = 50
    comment_ingest_interval_seconds: int = 600

    # ── NLP / Entity Extraction ──────────────────────────────────────────
    ner_model: str = "en_core_web_trf"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    topic_labels: List[str] = [
        "technology", "science", "gaming", "music", "sports", "politics",
        "education", "entertainment", "finance", "health", "cooking",
        "travel", "art", "nature", "history", "comedy", "news",
    ]
    entity_co_occurrence_window: int = 5
    min_entity_mentions: int = 2

    # ── Graph Visualization ──────────────────────────────────────────────
    graph_max_visible_nodes: int = 1000
    graph_sample_size: int = 500
    graph_event_buffer_size: int = 1000

    # ── Qdrant Collections ───────────────────────────────────────────────
    qdrant_collection_comments: str = "nexum_comments"
    comment_embedding_dim: int = 1024
    qdrant_collection_audio: str = "nexum_audio"
    audio_embedding_dim: int = 527

    # ── Audio Intelligence ───────────────────────────────────────────────
    audio_window_seconds: float = 5.0
    audio_hop_seconds: float = 2.5
    audio_sample_rate: int = 32000
    audio_min_event_confidence: float = 0.15
    audio_mel_bands: int = 128
    audio_source_labels: List[str] = ["music", "speech", "silence", "noise"]
    audio_event_labels: List[str] = [
        "applause", "laughter", "cheering", "crowd",
        "gunshot", "explosion", "siren", "alarm",
        "door_slam", "glass_breaking", "typing", "bell",
        "whistle", "crying", "screaming", "singing",
        "animal", "engine", "water", "wind",
        "thunder", "fireworks", "footsteps", "clapping",
    ]
    audio_speed_anomaly_threshold: float = 0.60
    audio_pitch_shift_threshold: float = 0.50
    audio_bpm_range: tuple = (30, 300)

    # ── Confidence Calibration (v4) ──────────────────────────────────────
    calibration_min_samples: int = 30
    calibration_auto_fit: bool = True

    # ── Audio-Transcript Alignment (v4) ──────────────────────────────────
    alignment_min_speech_prob: float = 0.3
    alignment_expected_wps: float = 2.5
    alignment_low_quality_threshold: float = 0.4

    # ── Acoustic Change Points (v4) ──────────────────────────────────────
    changepoint_spectral_threshold: float = 0.35
    changepoint_energy_ratio_threshold: float = 3.0
    changepoint_event_delta_threshold: float = 0.40
    changepoint_min_magnitude: float = 0.25
    changepoint_min_gap_seconds: float = 3.0

    # ── Paths ────────────────────────────────────────────────────────────
    config_dir: str = "/app/config"
    data_dir: str = "/app/data"
    temp_dir: str = "/tmp/nexum"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
