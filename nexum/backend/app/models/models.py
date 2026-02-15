"""
Nexum ORM Models — Complete data layer.

Every indexable entity becomes both a PostgreSQL row and a Neo4j graph node.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean, DateTime, Enum, Float, ForeignKey, Index,
    Integer, String, Text, func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# ═══════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════

class Platform(str, enum.Enum):
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    TWITCH = "twitch"
    DAILYMOTION = "dailymotion"
    VIMEO = "vimeo"
    REDDIT = "reddit"
    FACEBOOK = "facebook"
    BILIBILI = "bilibili"
    OTHER = "other"


class VideoStatus(str, enum.Enum):
    DISCOVERED = "discovered"
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    REINDEXING = "reindexing"


class CrawlPriority(str, enum.Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class FeedbackType(str, enum.Enum):
    UPVOTE = "upvote"
    DOWNVOTE = "downvote"
    TIMESTAMP_CORRECTION = "timestamp_correction"
    MISMATCH_REPORT = "mismatch_report"


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class CommentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSED = "processed"
    SPAM = "spam"
    TOXIC = "toxic"
    FAILED = "failed"


class EntityType(str, enum.Enum):
    PERSON = "person"
    ORGANIZATION = "org"
    LOCATION = "location"
    PRODUCT = "product"
    EVENT = "event"
    TECHNOLOGY = "technology"
    CONCEPT = "concept"
    OTHER = "other"


class LiveStatus(str, enum.Enum):
    NONE = "none"
    IS_LIVE = "is_live"
    WAS_LIVE = "was_live"
    IS_UPCOMING = "is_upcoming"
    POST_LIVE = "post_live"


# ═══════════════════════════════════════════════════════════════════════
# Core Content Models
# ═══════════════════════════════════════════════════════════════════════

class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), default=Platform.YOUTUBE)
    platform_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(256))
    url: Mapped[str] = mapped_column(String(512))
    custom_url: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    country: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    subscriber_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_videos: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    banner_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    priority_tier: Mapped[int] = mapped_column(Integer, default=1)
    last_crawled_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    videos: Mapped[List["Video"]] = relationship("Video", back_populates="channel", lazy="selectin")
    playlists: Mapped[List["Playlist"]] = relationship("Playlist", back_populates="channel", lazy="selectin")
    community_posts: Mapped[List["CommunityPost"]] = relationship("CommunityPost", back_populates="channel", lazy="selectin")


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        Index("ix_videos_status", "status"),
        Index("ix_videos_uploaded_at", "uploaded_at"),
        Index("ix_videos_platform", "platform"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), default=Platform.YOUTUBE)
    platform_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    channel_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("channels.id"), nullable=True)

    # Core metadata
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(1024))
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    duration_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    uploaded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Extended metadata
    tags: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    categories: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    live_status: Mapped[Optional[LiveStatus]] = mapped_column(Enum(LiveStatus), nullable=True)
    chapters_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    captions_info: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Engagement
    view_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    like_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dislike_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Content flags
    age_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_short: Mapped[bool] = mapped_column(Boolean, default=False)

    # Caption availability (summary — detail in Subtitle records)
    has_captions: Mapped[bool] = mapped_column(Boolean, default=False)
    caption_languages: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)
    has_auto_captions: Mapped[bool] = mapped_column(Boolean, default=False)
    auto_caption_languages: Mapped[Optional[list]] = mapped_column(ARRAY(String), nullable=True)

    # Processing
    status: Mapped[VideoStatus] = mapped_column(Enum(VideoStatus), default=VideoStatus.DISCOVERED)
    priority_score: Mapped[float] = mapped_column(Float, default=0.0)
    processing_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    channel: Mapped[Optional["Channel"]] = relationship("Channel", back_populates="videos")
    segments: Mapped[List["Segment"]] = relationship("Segment", back_populates="video", lazy="selectin", cascade="all, delete-orphan")
    frames: Mapped[List["Frame"]] = relationship("Frame", back_populates="video", lazy="selectin", cascade="all, delete-orphan")
    audio_segments: Mapped[List["AudioSegment"]] = relationship("AudioSegment", back_populates="video", lazy="selectin", cascade="all, delete-orphan")
    subtitles: Mapped[List["Subtitle"]] = relationship("Subtitle", back_populates="video", lazy="selectin", cascade="all, delete-orphan")
    chapters: Mapped[List["Chapter"]] = relationship("Chapter", back_populates="video", lazy="selectin", cascade="all, delete-orphan")
    stream_info: Mapped[List["StreamInfo"]] = relationship("StreamInfo", back_populates="video", lazy="selectin", cascade="all, delete-orphan")


class Playlist(Base):
    """Channel playlist — ordered collection of videos."""
    __tablename__ = "playlists"
    __table_args__ = (
        Index("ix_playlists_channel", "channel_id"),
        Index("ix_playlists_platform_pid", "platform_playlist_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_playlist_id: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    channel_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("channels.id"), nullable=True)
    title: Mapped[str] = mapped_column(String(512))
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    video_count: Mapped[int] = mapped_column(Integer, default=0)
    visibility: Mapped[str] = mapped_column(String(32), default="public")
    url: Mapped[str] = mapped_column(String(1024))
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    channel: Mapped[Optional["Channel"]] = relationship("Channel", back_populates="playlists")
    items: Mapped[List["PlaylistItem"]] = relationship("PlaylistItem", back_populates="playlist", lazy="selectin", cascade="all, delete-orphan")


class PlaylistItem(Base):
    """Junction: position of a video within a playlist."""
    __tablename__ = "playlist_items"
    __table_args__ = (
        Index("ix_pi_playlist_pos", "playlist_id", "position"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    playlist_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("playlists.id", ondelete="CASCADE"))
    video_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("videos.id", ondelete="SET NULL"), nullable=True)
    platform_video_id: Mapped[str] = mapped_column(String(128))
    position: Mapped[int] = mapped_column(Integer, default=0)
    added_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    playlist: Mapped["Playlist"] = relationship("Playlist", back_populates="items")


class CommunityPost(Base):
    """Channel community tab post (text, polls, images)."""
    __tablename__ = "community_posts"
    __table_args__ = (
        Index("ix_cp_channel", "channel_id"),
        Index("ix_cp_posted_at", "posted_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_post_id: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    post_type: Mapped[str] = mapped_column(String(32), default="text")
    image_urls: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    poll_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    posted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    channel: Mapped["Channel"] = relationship("Channel", back_populates="community_posts")


class Subtitle(Base):
    """Subtitle / caption track — auto or manual, per language."""
    __tablename__ = "subtitles"
    __table_args__ = (
        Index("ix_subtitles_video_lang", "video_id", "language"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    language: Mapped[str] = mapped_column(String(16))
    language_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_auto_generated: Mapped[bool] = mapped_column(Boolean, default=False)
    format: Mapped[str] = mapped_column(String(16), default="srt")
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    cue_count: Mapped[int] = mapped_column(Integer, default=0)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="subtitles")


class Chapter(Base):
    """Video chapter marker from YouTube chapters."""
    __tablename__ = "chapters"
    __table_args__ = (
        Index("ix_chapters_video", "video_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    title: Mapped[str] = mapped_column(String(512))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video: Mapped["Video"] = relationship("Video", back_populates="chapters")


class StreamInfo(Base):
    """Technical properties of a video/audio stream."""
    __tablename__ = "stream_info"
    __table_args__ = (
        Index("ix_si_video", "video_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    stream_type: Mapped[str] = mapped_column(String(16))
    codec: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    bitrate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    resolution: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    fps: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    container_format: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="stream_info")


# ═══════════════════════════════════════════════════════════════════════
# Transcript & Frames
# ═══════════════════════════════════════════════════════════════════════

class Segment(Base):
    __tablename__ = "segments"
    __table_args__ = (
        Index("ix_segments_video_start", "video_id", "start_time"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    speaker_label: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="segments")


class Frame(Base):
    __tablename__ = "frames"
    __table_args__ = (
        Index("ix_frames_video_ts", "video_id", "timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp: Mapped[float] = mapped_column(Float)
    visual_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    ocr_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    scene_label: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    is_scene_change: Mapped[bool] = mapped_column(Boolean, default=False)

    video: Mapped["Video"] = relationship("Video", back_populates="frames")


# ═══════════════════════════════════════════════════════════════════════
# Audio Intelligence
# ═══════════════════════════════════════════════════════════════════════

class AudioSegment(Base):
    __tablename__ = "audio_segments"
    __table_args__ = (
        Index("ix_audio_seg_video_ts", "video_id", "start_time"),
        Index("ix_audio_seg_source", "dominant_source"),
        Index("ix_audio_seg_bpm", "bpm"),
        Index("ix_audio_seg_music_prob", "music_probability"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    music_probability: Mapped[float] = mapped_column(Float, default=0.0)
    speech_probability: Mapped[float] = mapped_column(Float, default=0.0)
    dominant_source: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    source_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    event_tags: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    manipulation_scores: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    overall_manipulation: Mapped[float] = mapped_column(Float, default=0.0)
    bpm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    musical_key: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    loudness_lufs: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dynamic_range_db: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spectral_centroid: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spectral_rolloff: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    harmonic_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    zero_crossing_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    calibrated_manipulation: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    calibration_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    video: Mapped["Video"] = relationship("Video", back_populates="audio_segments")


class AcousticChangePoint(Base):
    __tablename__ = "acoustic_change_points"
    __table_args__ = (
        Index("ix_acp_video_ts", "video_id", "timestamp"),
        Index("ix_acp_magnitude", "magnitude"),
        Index("ix_acp_transition", "transition_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp: Mapped[float] = mapped_column(Float)
    magnitude: Mapped[float] = mapped_column(Float)
    transition_type: Mapped[str] = mapped_column(String(64))
    detail: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    from_state: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    to_state: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    spectral_distance: Mapped[float] = mapped_column(Float, default=0.0)
    energy_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video: Mapped["Video"] = relationship("Video", backref="change_points")


class TranscriptAlignment(Base):
    __tablename__ = "transcript_alignments"
    __table_args__ = (
        Index("ix_ta_video_ts", "video_id", "start_time"),
        Index("ix_ta_quality", "quality_level"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("segments.id", ondelete="SET NULL"), nullable=True)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    alignment_score: Mapped[float] = mapped_column(Float)
    quality_level: Mapped[str] = mapped_column(String(32))
    signals_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    warnings_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video: Mapped["Video"] = relationship("Video", backref="alignments")


# ═══════════════════════════════════════════════════════════════════════
# Users, Feedback & ML Admin
# ═══════════════════════════════════════════════════════════════════════

class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(256), unique=True)
    hashed_password: Mapped[str] = mapped_column(String(256))
    role: Mapped[UserRole] = mapped_column(Enum(UserRole), default=UserRole.USER)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("users.id"), nullable=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id"))
    segment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("segments.id"), nullable=True)
    query_text: Mapped[str] = mapped_column(Text)
    feedback_type: Mapped[FeedbackType] = mapped_column(Enum(FeedbackType))
    suggested_timestamp: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128))
    version: Mapped[str] = mapped_column(String(64))
    model_type: Mapped[str] = mapped_column(String(64))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)
    config_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    accuracy_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EvaluationMetric(Base):
    __tablename__ = "evaluation_metrics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name: Mapped[str] = mapped_column(String(128))
    metric_value: Mapped[float] = mapped_column(Float)
    model_version_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("model_versions.id"), nullable=True)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    measured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class CrawlLog(Base):
    __tablename__ = "crawl_logs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("channels.id"), nullable=True)
    video_platform_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    platform: Mapped[Optional[Platform]] = mapped_column(Enum(Platform), nullable=True)
    action: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# Social Knowledge Graph Models
# ═══════════════════════════════════════════════════════════════════════

class CommentAuthor(Base):
    """Public comment author — conversation participant."""
    __tablename__ = "comment_authors"
    __table_args__ = (
        Index("ix_comment_authors_display_name", "display_name"),
        Index("ix_comment_authors_platform_author", "platform", "platform_author_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), default=Platform.YOUTUBE)
    platform_author_id: Mapped[str] = mapped_column(String(256), index=True)
    display_name: Mapped[str] = mapped_column(String(256))
    comment_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    topic_vector: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    sentiment_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    entity_frequency_map: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    activity_histogram: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    is_pruned: Mapped[bool] = mapped_column(Boolean, default=False)

    comments: Mapped[List["Comment"]] = relationship("Comment", back_populates="author", lazy="selectin")


class Comment(Base):
    """Threaded comment node — recursive tree structure."""
    __tablename__ = "comments"
    __table_args__ = (
        Index("ix_comments_video_id", "video_id"),
        Index("ix_comments_parent_id", "parent_comment_id"),
        Index("ix_comments_root_thread", "root_thread_id"),
        Index("ix_comments_status", "status"),
        Index("ix_comments_depth", "depth_level"),
        Index("ix_comments_posted", "timestamp_posted"),
        Index("ix_comments_platform_cid", "platform_comment_id", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    platform_comment_id: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    author_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("comment_authors.id"), nullable=True)
    parent_comment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("comments.id"), nullable=True)
    root_thread_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), nullable=True)
    depth_level: Mapped[int] = mapped_column(Integer, default=0)
    child_count: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(Text)
    text_compressed: Mapped[Optional[bytes]] = mapped_column(nullable=True)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    embedding_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    sentiment_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sentiment_label: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    toxicity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spam_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extracted_timestamps: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    extracted_numbers: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    topic_labels: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    entities_json: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    status: Mapped[CommentStatus] = mapped_column(Enum(CommentStatus), default=CommentStatus.PENDING)
    timestamp_posted: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    video: Mapped["Video"] = relationship("Video", backref="comments")
    author: Mapped[Optional["CommentAuthor"]] = relationship("CommentAuthor", back_populates="comments")
    parent: Mapped[Optional["Comment"]] = relationship("Comment", remote_side="Comment.id", backref="children")


class Entity(Base):
    __tablename__ = "entities"
    __table_args__ = (
        Index("ix_entities_type", "entity_type"),
        Index("ix_entities_mention_count", "mention_count"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    canonical_name: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    entity_type: Mapped[EntityType] = mapped_column(Enum(EntityType), default=EntityType.OTHER)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    last_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    aliases: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    mentions: Mapped[List["EntityMention"]] = relationship("EntityMention", back_populates="entity", lazy="selectin")


class EntityMention(Base):
    __tablename__ = "entity_mentions"
    __table_args__ = (
        Index("ix_entity_mentions_entity", "entity_id"),
        Index("ix_entity_mentions_source", "source_type", "source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    video_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), nullable=True)
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(128))
    context_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    entity: Mapped["Entity"] = relationship("Entity", back_populates="mentions")


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    mention_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_vector: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TopicAssignment(Base):
    __tablename__ = "topic_assignments"
    __table_args__ = (
        Index("ix_topic_assign_topic", "topic_id"),
        Index("ix_topic_assign_source", "source_type", "source_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"))
    source_type: Mapped[str] = mapped_column(String(32))
    source_id: Mapped[str] = mapped_column(String(128))
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Recrawl Tracking
# ═══════════════════════════════════════════════════════════════════════

class RecrawlEvent(Base):
    """Tracks every recrawl of a video — what changed, when, why."""
    __tablename__ = "recrawl_events"
    __table_args__ = (
        Index("ix_recrawl_video", "video_id"),
        Index("ix_recrawl_trigger", "trigger"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    trigger: Mapped[str] = mapped_column(String(64))  # "manual" | "scheduled" | "drift_detected" | "input_duplicate"
    fields_changed: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # ["view_count","comment_count","description"]
    delta_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {"view_count": [old, new]}
    comments_added: Mapped[int] = mapped_column(Integer, default=0)
    comments_deleted: Mapped[int] = mapped_column(Integer, default=0)
    description_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    title_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Temporal-Visual Signals
# ═══════════════════════════════════════════════════════════════════════

class CursorHeatmap(Base):
    """Mouse/pointer movement tracking in screen recordings — where attention lingers."""
    __tablename__ = "cursor_heatmaps"
    __table_args__ = (Index("ix_cursor_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    grid_cells: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{x,y,dwell_ms,clicks}]
    hotspot_x: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # dominant focus point
    hotspot_y: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    total_distance_px: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    avg_velocity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    idle_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # fraction of time cursor is still
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TextEditEvent(Base):
    """Text typed then deleted/corrected in coding tutorials — the mistakes ARE the lesson."""
    __tablename__ = "text_edit_events"
    __table_args__ = (Index("ix_textedit_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp: Mapped[float] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String(32))  # "typed" | "deleted" | "corrected" | "pasted" | "undone"
    text_before: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    text_after: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    keystroke_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    time_to_correction_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_syntax_error_fix: Mapped[bool] = mapped_column(Boolean, default=False)
    is_logic_error_fix: Mapped[bool] = mapped_column(Boolean, default=False)
    programming_language: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class FaceCameraRatio(Base):
    """Face-to-camera presence ratio over time — when they stop showing their face, the content shifted."""
    __tablename__ = "face_camera_ratios"
    __table_args__ = (Index("ix_facecam_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    face_visible_ratio: Mapped[float] = mapped_column(Float)  # 0.0-1.0
    face_count: Mapped[int] = mapped_column(Integer, default=0)
    face_area_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # face area / frame area
    is_facecam_overlay: Mapped[bool] = mapped_column(Boolean, default=False)  # PiP webcam
    background_changed: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BackgroundChangeEvent(Base):
    """Background/studio changes between uploads — tracks life transitions."""
    __tablename__ = "background_changes"
    __table_args__ = (Index("ix_bgchange_channel", "channel_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    video_id_before: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    video_id_after: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    similarity_score: Mapped[float] = mapped_column(Float)  # 0=totally different, 1=same room
    change_type: Mapped[str] = mapped_column(String(64))  # "studio_upgrade" | "moved" | "lighting_change" | "equipment_change"
    embedding_distance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Audio Micro-Signals
# ═══════════════════════════════════════════════════════════════════════

class BreathingPattern(Base):
    """Breathing analysis — rushed breathing before controversial statements, deep breaths before rehearsed bits."""
    __tablename__ = "breathing_patterns"
    __table_args__ = (Index("ix_breath_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp: Mapped[float] = mapped_column(Float)
    breath_type: Mapped[str] = mapped_column(String(32))  # "deep_inhale" | "quick_gasp" | "sigh" | "held" | "panting"
    duration_ms: Mapped[int] = mapped_column(Integer)
    intensity: Mapped[float] = mapped_column(Float)  # 0-1
    context: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # "pre_statement" | "post_question" | "filler"
    preceding_speech_tempo: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # words/sec before breath
    following_speech_tempo: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AmbientAudioBleed(Base):
    """Keyboard clicks, mouse bleed, room noise — proves live coding vs pre-recorded playback."""
    __tablename__ = "ambient_audio_bleeds"
    __table_args__ = (Index("ix_ambient_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    bleed_type: Mapped[str] = mapped_column(String(64))  # "keyboard" | "mouse_click" | "chair_creak" | "fan_hum" | "notification"
    event_count: Mapped[int] = mapped_column(Integer, default=1)
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    implies_live_input: Mapped[bool] = mapped_column(Boolean, default=False)  # keyboard bleed = live coding
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class RoomAcousticFingerprint(Base):
    """Room reverb fingerprint — same creator filming in different locations across videos."""
    __tablename__ = "room_acoustic_fingerprints"
    __table_args__ = (Index("ix_roomfp_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    rt60_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # reverb time
    early_decay_time_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    clarity_c50: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    spectral_fingerprint: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # 32-dim vector
    room_size_estimate: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "small" | "medium" | "large" | "outdoor"
    noise_floor_db: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    matches_previous: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)  # same room as last video?
    cluster_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # groups same-room videos
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class LaughterEvent(Base):
    """Laughter authenticity score — genuine vs performative based on spectral decay."""
    __tablename__ = "laughter_events"
    __table_args__ = (Index("ix_laugh_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp: Mapped[float] = mapped_column(Float)
    duration_ms: Mapped[int] = mapped_column(Integer)
    authenticity_score: Mapped[float] = mapped_column(Float)  # 0=forced, 1=genuine
    spectral_decay_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pitch_variability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_audience: Mapped[bool] = mapped_column(Boolean, default=False)  # crowd vs. single person
    trigger_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # what was said before the laugh
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Behavioral Metadata
# ═══════════════════════════════════════════════════════════════════════

class UploadPattern(Base):
    """Upload time-of-day drift — a creator shifting from morning to 3AM is burning out."""
    __tablename__ = "upload_patterns"
    __table_args__ = (Index("ix_uploadpat_channel", "channel_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    upload_hour_utc: Mapped[int] = mapped_column(Integer)  # 0-23
    upload_day_of_week: Mapped[int] = mapped_column(Integer)  # 0=Mon, 6=Sun
    days_since_last_upload: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hour_drift_from_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # stddevs from channel mean
    is_anomalous_time: Mapped[bool] = mapped_column(Boolean, default=False)  # >2σ from pattern
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TitleEditHistory(Base):
    """Title/thumbnail edit history — what they A/B tested, what they settled on."""
    __tablename__ = "title_edit_history"
    __table_args__ = (Index("ix_titleedit_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    field_name: Mapped[str] = mapped_column(String(32))  # "title" | "thumbnail" | "description" | "tags"
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    hours_after_upload: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ThumbnailAnalysis(Base):
    """Thumbnail color palette evolution — tracks brand maturity per channel."""
    __tablename__ = "thumbnail_analyses"
    __table_args__ = (Index("ix_thumb_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    channel_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("channels.id"), nullable=True)
    dominant_colors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{"hex":"#ff0000","pct":0.35}]
    has_face: Mapped[bool] = mapped_column(Boolean, default=False)
    has_text_overlay: Mapped[bool] = mapped_column(Boolean, default=False)
    text_content: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)  # OCR of thumbnail text
    clickbait_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # 0-1
    palette_distance_from_channel_mean: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    brightness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    saturation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    contrast: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DescriptionLinkHealth(Base):
    """Description link rot rate — how many linked resources are now 404."""
    __tablename__ = "description_link_health"
    __table_args__ = (Index("ix_linkhealth_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    url: Mapped[str] = mapped_column(String(2048))
    domain: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_alive: Mapped[bool] = mapped_column(Boolean, default=True)
    is_affiliate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_self_reference: Mapped[bool] = mapped_column(Boolean, default=False)  # links to own channel/video
    last_checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    first_dead_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Comment Archaeology
# ═══════════════════════════════════════════════════════════════════════

class PinnedCommentHistory(Base):
    """Pinned comment changes over time — what the creator wanted highlighted shifted."""
    __tablename__ = "pinned_comment_history"
    __table_args__ = (Index("ix_pinned_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    comment_id: Mapped[Optional[uuid.UUID]] = mapped_column(ForeignKey("comments.id", ondelete="SET NULL"), nullable=True)
    pinned_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pinned_by_creator: Mapped[bool] = mapped_column(Boolean, default=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    unpinned_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class CommentSentimentDrift(Base):
    """First-hour comment sentiment vs day-30 — algorithmic push changes who finds it."""
    __tablename__ = "comment_sentiment_drift"
    __table_args__ = (Index("ix_sentdrift_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    window_label: Mapped[str] = mapped_column(String(32))  # "hour_1" | "day_1" | "day_7" | "day_30" | "day_90"
    avg_sentiment: Mapped[float] = mapped_column(Float)
    comment_count: Mapped[int] = mapped_column(Integer)
    positive_ratio: Mapped[float] = mapped_column(Float)
    negative_ratio: Mapped[float] = mapped_column(Float)
    toxicity_avg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    language_distribution: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {"en":0.6,"ar":0.3}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class DeletedCommentShadow(Base):
    """Reply chains that respond to something now missing — deleted comment shadows."""
    __tablename__ = "deleted_comment_shadows"
    __table_args__ = (Index("ix_shadow_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    platform_comment_id: Mapped[str] = mapped_column(String(256))  # the deleted comment's platform ID
    orphaned_reply_count: Mapped[int] = mapped_column(Integer, default=0)
    last_known_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # if we captured it before deletion
    last_known_author: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    probable_reason: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # "creator_deleted" | "spam_filter" | "author_deleted"
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Cross-Video Forensics
# ═══════════════════════════════════════════════════════════════════════

class BRollReuse(Base):
    """B-roll reuse fingerprinting — same stock footage across creators reveals content mills."""
    __tablename__ = "broll_reuse"
    __table_args__ = (Index("ix_broll_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    matched_video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    timestamp_source: Mapped[float] = mapped_column(Float)
    timestamp_match: Mapped[float] = mapped_column(Float)
    duration_seconds: Mapped[float] = mapped_column(Float)
    similarity_score: Mapped[float] = mapped_column(Float)  # perceptual hash similarity
    is_stock_footage: Mapped[bool] = mapped_column(Boolean, default=False)
    fingerprint_hash: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class SponsorSegment(Base):
    """Sponsor segment cadence — exact second sponsors appear, tracked across the ecosystem."""
    __tablename__ = "sponsor_segments"
    __table_args__ = (Index("ix_sponsor_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    sponsor_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    segment_type: Mapped[str] = mapped_column(String(32))  # "sponsor" | "self_promo" | "interaction_reminder" | "merch"
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    position_ratio: Mapped[float] = mapped_column(Float)  # 0.0=start, 0.5=mid, 1.0=end
    duration_seconds: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class IntroOutroEvolution(Base):
    """Intro/outro evolution — when a creator drops their intro, something changed strategically."""
    __tablename__ = "intro_outro_evolutions"
    __table_args__ = (Index("ix_introoutro_channel", "channel_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    has_intro: Mapped[bool] = mapped_column(Boolean, default=False)
    intro_duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    intro_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)  # audio hash
    has_outro: Mapped[bool] = mapped_column(Boolean, default=False)
    outro_duration_s: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    outro_fingerprint: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    silence_before_ask_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # pause before "subscribe"
    ask_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # "subscribe" | "like" | "comment" | "bell" | "none"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Linguistic Signals
# ═══════════════════════════════════════════════════════════════════════

class LinguisticProfile(Base):
    """Per-video linguistic analysis — code-switching, filler words, hedging, vocabulary."""
    __tablename__ = "linguistic_profiles"
    __table_args__ = (Index("ix_lingprof_video", "video_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    # Code-switching
    code_switch_count: Mapped[int] = mapped_column(Integer, default=0)
    code_switch_pairs: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{"from":"en","to":"ar","timestamp":12.5}]
    primary_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    secondary_language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    language_ratio: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {"en":0.7,"ar":0.3}
    # Filler words
    filler_word_count: Mapped[int] = mapped_column(Integer, default=0)
    filler_word_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # per minute
    filler_clusters: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{timestamp, density}] — spikes at knowledge boundaries
    # Hedging
    hedging_phrases_count: Mapped[int] = mapped_column(Integer, default=0)
    hedging_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # per 100 words
    hedging_by_topic: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # {"quantum_physics":0.12,"python":0.02}
    # Vocabulary
    unique_word_count: Mapped[int] = mapped_column(Integer, default=0)
    vocabulary_richness: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # type-token ratio
    zipf_deviation: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # deviation from expected Zipf distribution
    rare_word_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # words outside top 5000 per minute
    avg_sentence_length: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    speaking_rate_wpm: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ═══════════════════════════════════════════════════════════════════════
# v4.2: Graph-Relational Signals
# ═══════════════════════════════════════════════════════════════════════

class CommenterMigration(Base):
    """Commenter migration patterns — users who shift from Channel A to Channel B."""
    __tablename__ = "commenter_migrations"
    __table_args__ = (
        Index("ix_migration_author", "author_id"),
        Index("ix_migration_channels", "from_channel_id", "to_channel_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("comment_authors.id", ondelete="CASCADE"))
    from_channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    to_channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    first_seen_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_seen_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    first_seen_to: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    migration_confidence: Mapped[float] = mapped_column(Float)  # 0-1
    overlap_period_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # days active on both
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class EntityPropagation(Base):
    """Entity first-mention attribution — which channel said 'Qdrant' first, and how it propagated."""
    __tablename__ = "entity_propagations"
    __table_args__ = (
        Index("ix_entprop_entity", "entity_id"),
        Index("ix_entprop_origin_channel", "origin_channel_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    origin_channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    origin_video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    origin_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    propagation_chain: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [{channel_id, video_id, timestamp, lag_hours}]
    total_adopters: Mapped[int] = mapped_column(Integer, default=0)
    avg_adoption_lag_hours: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ParasocialIndex(Base):
    """Parasocial reciprocity — ratio of creator replies to a specific commenter vs that commenter's activity."""
    __tablename__ = "parasocial_indices"
    __table_args__ = (
        Index("ix_parasocial_author", "author_id"),
        Index("ix_parasocial_channel", "channel_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    author_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("comment_authors.id", ondelete="CASCADE"))
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    author_comments_on_channel: Mapped[int] = mapped_column(Integer, default=0)
    creator_replies_to_author: Mapped[int] = mapped_column(Integer, default=0)
    reciprocity_ratio: Mapped[float] = mapped_column(Float, default=0.0)  # replies/comments
    first_interaction: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    last_interaction: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    engagement_tier: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)  # "superfan" | "regular" | "casual"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class TopicGestation(Base):
    """Topic gestation period — time between entity first appearing in comments and creator making a video about it."""
    __tablename__ = "topic_gestations"
    __table_args__ = (
        Index("ix_topicgest_channel", "channel_id"),
        Index("ix_topicgest_entity", "entity_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"))
    entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("entities.id", ondelete="CASCADE"))
    first_comment_mention_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    first_video_mention_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    gestation_days: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    comment_mention_count_before_video: Mapped[int] = mapped_column(Integer, default=0)
    audience_requested: Mapped[bool] = mapped_column(Boolean, default=False)  # comments explicitly asked for it
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
