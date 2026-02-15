"""
Nexum API Schemas — Pydantic v2 models for request/response validation.

Single source of truth — no v1/v3/v4 duplication.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ═══════════════════════════════════════════════════════════════════════
# Search
# ═══════════════════════════════════════════════════════════════════════

class SearchFilters(BaseModel):
    min_duration: Optional[int] = None
    max_duration: Optional[int] = None
    min_views: Optional[int] = None
    max_views: Optional[int] = None
    upload_after: Optional[datetime] = None
    upload_before: Optional[datetime] = None
    channels: Optional[List[str]] = None
    language: Optional[str] = None
    tags: Optional[List[str]] = None
    modality: Optional[str] = None
    min_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    platform: Optional[str] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1024)
    filters: Optional[SearchFilters] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)


class ModalityScore(BaseModel):
    """All 14 fusion modalities in one schema."""
    text_semantic: float = 0.0
    visual_similarity: float = 0.0
    ocr_match: float = 0.0
    keyword_match: float = 0.0
    temporal_coherence: float = 0.0
    emotion_context: float = 0.0
    comment_semantic: float = 0.0
    comment_timestamp_boost: float = 0.0
    agreement_cluster: float = 0.0
    entity_overlap: float = 0.0
    user_cluster_confidence: float = 0.0
    audio_event_match: float = 0.0
    audio_attribute_match: float = 0.0
    alignment_quality: float = 0.0
    change_point_proximity: float = 0.0


class SearchResult(BaseModel):
    video_id: str
    video_title: str
    channel_name: Optional[str] = None
    timestamp: float
    end_timestamp: Optional[float] = None
    transcript_snippet: Optional[str] = None
    visual_tags: Optional[List[str]] = None
    ocr_text: Optional[str] = None
    confidence_score: float
    modality_scores: ModalityScore
    video_url: str
    thumbnail_url: Optional[str] = None
    view_count: Optional[int] = None
    explanation: str
    video_url_with_timecode: str


class QueryDecomposition(BaseModel):
    objects: List[str] = []
    actions: List[str] = []
    numbers: List[str] = []
    time_era: Optional[str] = None
    emotion: Optional[str] = None
    context: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    total_results: int
    page: int
    page_size: int
    results: List[SearchResult]
    query_decomposition: Optional[QueryDecomposition] = None
    latency_ms: float


# ═══════════════════════════════════════════════════════════════════════
# Video
# ═══════════════════════════════════════════════════════════════════════

class SegmentSchema(BaseModel):
    id: str
    start_time: float
    end_time: float
    text: str
    confidence: float
    speaker_label: Optional[str] = None


class FrameSchema(BaseModel):
    id: str
    timestamp: float
    visual_tags: Optional[List[dict]] = None
    ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    scene_label: Optional[str] = None
    is_scene_change: bool = False


class SubtitleSchema(BaseModel):
    id: str
    language: str
    language_name: Optional[str] = None
    is_auto_generated: bool = False
    format: str = "srt"
    cue_count: int = 0


class StreamInfoSchema(BaseModel):
    stream_type: str
    codec: Optional[str] = None
    bitrate: Optional[int] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    container_format: Optional[str] = None
    file_size_bytes: Optional[int] = None


class ChapterSchema(BaseModel):
    title: str
    start_time: float
    end_time: Optional[float] = None


class VideoSummary(BaseModel):
    id: str
    platform: str
    platform_id: str
    title: str
    channel_name: Optional[str] = None
    url: str
    thumbnail_url: Optional[str] = None
    duration_seconds: Optional[int] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    tags: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    live_status: Optional[str] = None
    status: str
    segment_count: int = 0
    frame_count: int = 0
    created_at: datetime


class VideoDetail(VideoSummary):
    description: Optional[str] = None
    language: Optional[str] = None
    chapters: Optional[List[dict]] = None
    captions_info: Optional[dict] = None
    segments: List[SegmentSchema] = []
    frames: List[FrameSchema] = []
    subtitles: List[SubtitleSchema] = []
    streams: List[StreamInfoSchema] = []


# ═══════════════════════════════════════════════════════════════════════
# Playlist / Channel / Community
# ═══════════════════════════════════════════════════════════════════════

class PlaylistSchema(BaseModel):
    id: str
    platform_playlist_id: str
    title: str
    description: Optional[str] = None
    thumbnail_url: Optional[str] = None
    video_count: int = 0
    url: str


class PlaylistItemSchema(BaseModel):
    platform_video_id: str
    position: int = 0
    video_id: Optional[str] = None


class ChannelDetailSchema(BaseModel):
    id: str
    platform: str
    platform_id: str
    name: str
    url: str
    custom_url: Optional[str] = None
    description: Optional[str] = None
    country: Optional[str] = None
    subscriber_count: Optional[int] = None
    total_videos: Optional[int] = None
    banner_url: Optional[str] = None
    video_count: int = 0
    playlist_count: int = 0


class CommunityPostSchema(BaseModel):
    id: str
    platform_post_id: str
    text: Optional[str] = None
    post_type: str = "text"
    like_count: int = 0
    comment_count: int = 0
    posted_at: Optional[datetime] = None


# ═══════════════════════════════════════════════════════════════════════
# Feedback
# ═══════════════════════════════════════════════════════════════════════

class FeedbackCreate(BaseModel):
    video_id: str
    segment_id: Optional[str] = None
    query_text: str
    feedback_type: str
    suggested_timestamp: Optional[float] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: str
    status: str = "received"


# ═══════════════════════════════════════════════════════════════════════
# Admin
# ═══════════════════════════════════════════════════════════════════════

class FusionWeightsUpdate(BaseModel):
    """All 14 modality weights — omit fields to leave unchanged."""
    weight_text_semantic: Optional[float] = None
    weight_visual_similarity: Optional[float] = None
    weight_ocr_match: Optional[float] = None
    weight_keyword_match: Optional[float] = None
    weight_temporal_coherence: Optional[float] = None
    weight_emotion_context: Optional[float] = None
    weight_comment_semantic: Optional[float] = None
    weight_comment_timestamp_boost: Optional[float] = None
    weight_agreement_cluster: Optional[float] = None
    weight_entity_overlap: Optional[float] = None
    weight_user_cluster_confidence: Optional[float] = None
    weight_audio_event_match: Optional[float] = None
    weight_audio_attribute_match: Optional[float] = None
    weight_alignment_quality: Optional[float] = None
    weight_change_point_proximity: Optional[float] = None


class PriorityRuleUpdate(BaseModel):
    channel_id: Optional[str] = None
    priority_tier: Optional[int] = None
    min_views: Optional[int] = None
    keyword_boosts: Optional[dict] = None
    weight_views: Optional[float] = None
    weight_recency: Optional[float] = None


class ReindexRequest(BaseModel):
    video_ids: Optional[List[str]] = None
    model_version: Optional[str] = None
    force: bool = False


class SystemMetrics(BaseModel):
    total_videos: int
    indexed_videos: int
    queued_videos: int
    failed_videos: int
    total_segments: int
    total_frames: int
    total_feedback: int
    upvote_ratio: float
    avg_search_latency_ms: float
    worker_count: int
    queue_depth: int
    storage_used_gb: float
    # Graph stats
    total_comments: int = 0
    total_entities: int = 0
    total_comment_authors: int = 0
    total_topics: int = 0
    total_playlists: int = 0
    total_community_posts: int = 0
    graph_edges: int = 0
    # Audio stats
    total_audio_segments: int = 0
    total_change_points: int = 0
    total_alignment_warnings: int = 0
    avg_alignment_score: float = 0.0
    calibration_version: str = "uncalibrated"


class ModelVersionSchema(BaseModel):
    id: str
    name: str
    version: str
    model_type: str
    is_active: bool
    accuracy_metrics: Optional[dict] = None
    created_at: datetime


class EvaluationMetricSchema(BaseModel):
    metric_name: str
    metric_value: float
    model_version: Optional[str] = None
    measured_at: datetime


# ═══════════════════════════════════════════════════════════════════════
# Comments & Social Graph
# ═══════════════════════════════════════════════════════════════════════

class CommentSchema(BaseModel):
    id: str
    platform_comment_id: str
    video_id: str
    author_display_name: Optional[str] = None
    parent_comment_id: Optional[str] = None
    root_thread_id: Optional[str] = None
    depth_level: int = 0
    child_count: int = 0
    text: str
    like_count: int = 0
    reply_count: int = 0
    language: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    toxicity_score: Optional[float] = None
    topic_labels: Optional[List[str]] = None
    entities: Optional[List[str]] = None
    extracted_timestamps: Optional[List[str]] = None
    status: str = "pending"
    timestamp_posted: Optional[datetime] = None


class CommentThreadSchema(BaseModel):
    root: CommentSchema
    children: List["CommentThreadSchema"] = []
    total_depth: int = 0
    total_comments: int = 0


class CommentIngestRequest(BaseModel):
    video_id: str
    max_comments: int = Field(200, ge=1, le=5000)
    include_replies: bool = True
    max_reply_depth: int = Field(20, ge=0, le=50)


class CommentIngestResponse(BaseModel):
    video_id: str
    comments_fetched: int
    comments_new: int
    comments_spam_filtered: int
    authors_discovered: int
    status: str = "completed"


class CommentAuthorSchema(BaseModel):
    id: str
    display_name: str
    platform: str
    comment_count: int = 0
    first_seen_at: Optional[datetime] = None
    last_seen_at: Optional[datetime] = None
    top_topics: Optional[List[str]] = None
    sentiment_distribution: Optional[dict] = None


class EntitySchema(BaseModel):
    id: str
    canonical_name: str
    entity_type: str
    mention_count: int = 0
    aliases: Optional[List[str]] = None
    first_seen_at: Optional[datetime] = None


class EntityMentionSchema(BaseModel):
    entity_name: str
    entity_type: str
    source_type: str
    source_id: str
    confidence: float = 1.0


class TopicSchema(BaseModel):
    id: str
    name: str
    mention_count: int = 0
    description: Optional[str] = None


# ═══════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════

class GraphNodeSchema(BaseModel):
    id: str
    label: str
    node_type: str
    data: Optional[dict] = None


class GraphEdgeSchema(BaseModel):
    source: str
    target: str
    edge_type: str
    weight: float = 1.0
    data: Optional[dict] = None


class GraphSnapshotSchema(BaseModel):
    nodes: List[GraphNodeSchema]
    edges: List[GraphEdgeSchema]
    total_nodes: int
    total_edges: int
    sampled: bool = False


class GraphTraversalRequest(BaseModel):
    start_node_id: str
    start_node_type: str = "Comment"
    direction: str = "both"
    max_depth: int = Field(3, ge=1, le=10)
    edge_types: Optional[List[str]] = None
    limit: int = Field(200, ge=1, le=1000)


class ThreadAnalysis(BaseModel):
    root_comment_id: str
    total_depth: int
    total_comments: int
    unique_authors: int
    avg_sentiment: float
    debate_score: float
    longest_chain_length: int
    most_branched_depth: int


class GraphStats(BaseModel):
    video_count: int = 0
    comment_count: int = 0
    comment_author_count: int = 0
    entity_count: int = 0
    topic_count: int = 0
    channel_count: int = 0
    playlist_count: int = 0
    total_edges: int = 0
    comments_pending: int = 0
    comments_processed: int = 0
    comments_spam: int = 0
    avg_thread_depth: float = 0.0
    ingest_rate_per_min: float = 0.0


# ═══════════════════════════════════════════════════════════════════════
# Audio Intelligence
# ═══════════════════════════════════════════════════════════════════════

class AudioEventSchema(BaseModel):
    label: str
    confidence: float
    category: str = "event"


class ManipulationScoresSchema(BaseModel):
    speed_anomaly: float = 0.0
    pitch_shift: float = 0.0
    compression: float = 0.0
    reverb_echo: float = 0.0
    robotic_autotune: float = 0.0
    time_stretch: float = 0.0
    overall_manipulation: float = 0.0


class MusicalAttributesSchema(BaseModel):
    bpm: Optional[float] = None
    key: Optional[str] = None
    loudness_lufs: float = -30.0
    dynamic_range_db: float = 0.0
    spectral_centroid: float = 0.0
    spectral_rolloff: float = 0.0
    harmonic_ratio: float = 0.0
    zero_crossing_rate: float = 0.0


class AudioSegmentSchema(BaseModel):
    id: str
    video_id: str
    start_time: float
    end_time: float
    music_probability: float = 0.0
    speech_probability: float = 0.0
    dominant_source: Optional[str] = None
    source_tags: Optional[List[AudioEventSchema]] = None
    event_tags: Optional[List[AudioEventSchema]] = None
    manipulation_scores: Optional[ManipulationScoresSchema] = None
    overall_manipulation: float = 0.0
    bpm: Optional[float] = None
    musical_key: Optional[str] = None
    loudness_lufs: Optional[float] = None
    dynamic_range_db: Optional[float] = None
    spectral_centroid: Optional[float] = None
    harmonic_ratio: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class AudioAnalysisSummary(BaseModel):
    video_id: str
    duration_seconds: float
    total_windows: int
    global_bpm: Optional[float] = None
    global_key: Optional[str] = None
    dominant_source: str = "unknown"
    total_music_seconds: float = 0.0
    total_speech_seconds: float = 0.0
    total_silence_seconds: float = 0.0
    manipulation_summary: Optional[ManipulationScoresSchema] = None


class AudioSearchHint(BaseModel):
    has_music: bool = False
    has_speech: bool = False
    dominant_events: List[str] = []
    avg_bpm: Optional[float] = None
    manipulation_flag: bool = False


class AudioStatsSchema(BaseModel):
    total_audio_segments: int = 0
    videos_with_audio_analysis: int = 0
    avg_music_ratio: float = 0.0
    avg_speech_ratio: float = 0.0
    manipulation_flagged_count: int = 0
    bpm_distribution: Optional[Dict[str, int]] = None


# ═══════════════════════════════════════════════════════════════════════
# Confidence Calibration
# ═══════════════════════════════════════════════════════════════════════

class CalibratedScoreSchema(BaseModel):
    raw_score: float
    calibrated_probability: float
    band: str
    band_label: str
    band_color: str
    model_name: str
    calibration_version: str


class CalibrationStatusSchema(BaseModel):
    version: str
    targets: Dict[str, dict]


# ═══════════════════════════════════════════════════════════════════════
# Audio-Transcript Alignment
# ═══════════════════════════════════════════════════════════════════════

class AlignmentSignalSchema(BaseModel):
    name: str
    score: float
    detail: str = ""


class AlignmentResultSchema(BaseModel):
    start_time: float
    end_time: float
    alignment_score: float
    quality_level: str
    signals: List[AlignmentSignalSchema] = []
    warnings: List[str] = []


class AlignmentSummarySchema(BaseModel):
    video_id: str
    overall_score: float
    overall_quality: str
    total_warnings: int
    mismatch_regions: List[List[float]] = []
    segments: List[AlignmentResultSchema] = []


# ═══════════════════════════════════════════════════════════════════════
# Acoustic Change Points
# ═══════════════════════════════════════════════════════════════════════

class ChangePointSchema(BaseModel):
    timestamp: float
    magnitude: float
    transition_type: str
    detail: str = ""
    from_state: str = ""
    to_state: str = ""


class ChangePointResultSchema(BaseModel):
    video_id: str
    total_change_points: int
    total_duration: float
    num_scenes: int
    dominant_transitions: List[str] = []
    change_points: List[ChangePointSchema] = []
