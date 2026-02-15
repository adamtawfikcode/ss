"""
Nexum API — Video routes.
"""
from __future__ import annotations

import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.models import Channel, Frame, Segment, Subtitle, StreamInfo, Video
from app.schemas.schemas import FrameSchema, SegmentSchema, SubtitleSchema, StreamInfoSchema, VideoDetail, VideoSummary

router = APIRouter(prefix="/videos", tags=["Videos"])


@router.get("", response_model=List[VideoSummary])
async def list_videos(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    status: str | None = None,
    platform: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List indexed videos with pagination."""
    query = select(Video).order_by(Video.created_at.desc())
    if status:
        query = query.where(Video.status == status)
    if platform:
        query = query.where(Video.platform == platform)
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    videos = result.scalars().all()

    if not videos:
        return []

    # Batch fetch counts and channel names — eliminates N+1
    video_ids = [v.id for v in videos]
    channel_ids = [v.channel_id for v in videos if v.channel_id]

    seg_counts = {}
    frame_counts = {}
    ch_names = {}

    # Segment counts in one query
    seg_result = await db.execute(
        select(Segment.video_id, func.count(Segment.id))
        .where(Segment.video_id.in_(video_ids))
        .group_by(Segment.video_id)
    )
    for vid, cnt in seg_result:
        seg_counts[vid] = cnt

    # Frame counts in one query
    frame_result = await db.execute(
        select(Frame.video_id, func.count(Frame.id))
        .where(Frame.video_id.in_(video_ids))
        .group_by(Frame.video_id)
    )
    for vid, cnt in frame_result:
        frame_counts[vid] = cnt

    # Channel names in one query
    if channel_ids:
        ch_result = await db.execute(
            select(Channel.id, Channel.name).where(Channel.id.in_(channel_ids))
        )
        for cid, cname in ch_result:
            ch_names[cid] = cname

    summaries = []
    for v in videos:
        summaries.append(VideoSummary(
            id=str(v.id),
            platform=v.platform.value if v.platform else "other",
            platform_id=v.platform_id,
            title=v.title,
            channel_name=ch_names.get(v.channel_id),
            url=v.url,
            thumbnail_url=v.thumbnail_url,
            duration_seconds=v.duration_seconds,
            view_count=v.view_count,
            like_count=v.like_count,
            comment_count=v.comment_count,
            tags=v.tags,
            categories=v.categories,
            live_status=v.live_status.value if v.live_status else None,
            status=v.status.value,
            segment_count=seg_counts.get(v.id, 0),
            frame_count=frame_counts.get(v.id, 0),
            created_at=v.created_at,
        ))

    return summaries


@router.get("/{video_id}", response_model=VideoDetail)
async def get_video(video_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed video information including segments and frames."""
    try:
        vid_uuid = uuid.UUID(video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video ID format")

    video = await db.get(Video, vid_uuid)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Fetch segments
    seg_result = await db.execute(
        select(Segment).where(Segment.video_id == vid_uuid).order_by(Segment.start_time)
    )
    segments = seg_result.scalars().all()

    # Fetch frames
    frame_result = await db.execute(
        select(Frame).where(Frame.video_id == vid_uuid).order_by(Frame.timestamp)
    )
    frames = frame_result.scalars().all()

    # Fetch subtitles
    sub_result = await db.execute(
        select(Subtitle).where(Subtitle.video_id == vid_uuid)
    )
    subtitles = sub_result.scalars().all()

    # Fetch stream info
    stream_result = await db.execute(
        select(StreamInfo).where(StreamInfo.video_id == vid_uuid)
    )
    streams = stream_result.scalars().all()

    ch_name = None
    if video.channel_id:
        ch_name = await db.scalar(select(Channel.name).where(Channel.id == video.channel_id))

    return VideoDetail(
        id=str(video.id),
        platform=video.platform.value if video.platform else "other",
        platform_id=video.platform_id,
        title=video.title,
        channel_name=ch_name,
        url=video.url,
        thumbnail_url=video.thumbnail_url,
        duration_seconds=video.duration_seconds,
        view_count=video.view_count,
        like_count=video.like_count,
        comment_count=video.comment_count,
        tags=video.tags,
        categories=video.categories,
        live_status=video.live_status.value if video.live_status else None,
        status=video.status.value,
        description=video.description,
        language=video.language,
        chapters=video.chapters_json,
        captions_info=video.captions_info,
        segment_count=len(segments),
        frame_count=len(frames),
        created_at=video.created_at,
        segments=[
            SegmentSchema(
                id=str(s.id),
                start_time=s.start_time,
                end_time=s.end_time,
                text=s.text,
                confidence=s.confidence,
                speaker_label=s.speaker_label,
            )
            for s in segments
        ],
        frames=[
            FrameSchema(
                id=str(f.id),
                timestamp=f.timestamp,
                visual_tags=f.visual_tags,
                ocr_text=f.ocr_text,
                ocr_confidence=f.ocr_confidence,
                scene_label=f.scene_label,
                is_scene_change=f.is_scene_change,
            )
            for f in frames
        ],
        subtitles=[
            SubtitleSchema(
                id=str(sub.id),
                language=sub.language,
                language_name=sub.language_name,
                is_auto_generated=sub.is_auto_generated,
                format=sub.format,
                cue_count=sub.cue_count,
            )
            for sub in subtitles
        ],
        streams=[
            StreamInfoSchema(
                stream_type=si.stream_type,
                codec=si.codec,
                bitrate=si.bitrate,
                resolution=si.resolution,
                fps=si.fps,
                sample_rate=si.sample_rate,
                channels=si.channels,
                container_format=si.container_format,
                file_size_bytes=si.file_size_bytes,
            )
            for si in streams
        ],
    )
