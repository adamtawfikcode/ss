"""
Nexum Media Processing Service — full pipeline for video ingestion.

Quality-first, sequential GPU processing for 12 GB VRAM (RTX 5070 Ti).

Pipeline (models loaded/released sequentially):
 1. Download video (yt-dlp — YouTube, TikTok, Twitter, 1700+ sites)
 2. Extract audio → WAV 16 kHz mono
 3. [GPU: load Whisper large-v3] Transcribe → chunk segments with overlap
 4. [GPU: release Whisper]
 5. Extract frames (1.5 s interval, scene change detection, blur filter)
 6. [GPU: load CLIP ViT-H-14] OCR dual-engine + visual classification + embeddings
 7. Store in PostgreSQL + Qdrant
 8. [GPU: release CLIP]
 9. Store subtitles + stream info from yt-dlp metadata
 9. Cleanup temp files

Each video is processed once — quality over speed.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.database import async_session_factory
from app.ml.embeddings.embedding_service import embedding_service
from app.ml.ocr.ocr_service import ocr_service, MergedOCRResult
from app.ml.speech.speech_service import speech_service
from app.ml.vision.vision_service import vision_service
from app.ml.audio.audio_intelligence_service import audio_intelligence_service
from app.ml.audio.alignment_service import alignment_service
from app.ml.audio.change_points import change_point_detector
from app.ml.calibration import calibration_service
from app.models.models import (
    AcousticChangePoint, AudioSegment, Chapter, Frame, LiveStatus, Platform,
    Segment, StreamInfo, Subtitle, TranscriptAlignment, Video, VideoStatus,
)
from app.services.search.vector_store import vector_store

logger = logging.getLogger(__name__)
settings = get_settings()


class MediaProcessingService:
    """Orchestrates the full video processing pipeline."""

    # ── Platform Detection ───────────────────────────────────────────────

    @staticmethod
    def detect_platform(url: str) -> Platform:
        """Detect which platform a URL belongs to."""
        url_lower = url.lower()
        patterns = {
            Platform.YOUTUBE: [r"youtube\.com", r"youtu\.be", r"youtube\.com/shorts"],
            Platform.TIKTOK: [r"tiktok\.com", r"vm\.tiktok\.com"],
            Platform.TWITTER: [r"twitter\.com", r"x\.com", r"t\.co"],
            Platform.INSTAGRAM: [r"instagram\.com"],
            Platform.TWITCH: [r"twitch\.tv", r"clips\.twitch\.tv"],
            Platform.DAILYMOTION: [r"dailymotion\.com", r"dai\.ly"],
            Platform.VIMEO: [r"vimeo\.com"],
            Platform.REDDIT: [r"reddit\.com", r"v\.redd\.it"],
            Platform.FACEBOOK: [r"facebook\.com", r"fb\.watch"],
            Platform.BILIBILI: [r"bilibili\.com", r"b23\.tv"],
        }
        for platform, regexes in patterns.items():
            for regex in regexes:
                if re.search(regex, url_lower):
                    return platform
        return Platform.OTHER

    # ── Main Entry Point ─────────────────────────────────────────────────

    async def process_video(self, video_id: str):
        """Main entry point — processes a single video end-to-end."""
        async with async_session_factory() as db:
            video = await db.get(Video, video_id)
            if not video:
                logger.error(f"Video not found: {video_id}")
                return

            # Transition: QUEUED → DOWNLOADING
            video.status = VideoStatus.DOWNLOADING
            await db.commit()

            work_dir = Path(tempfile.mkdtemp(prefix="nexum_"))
            try:
                # ── Phase 1: Download ────────────────────────────────
                media_path = await self._download_video(video, work_dir)
                if not media_path:
                    raise RuntimeError("Download failed — no media file produced")

                # ── Phase 1b: Extract full metadata ──────────────────
                await self._enrich_video_metadata(db, video, work_dir, media_path)

                # Transition: DOWNLOADING → PROCESSING
                video.status = VideoStatus.PROCESSING
                await db.commit()

                # ── Phase 2: Audio + Transcription (Whisper on GPU) ──
                audio_path = self._extract_audio(media_path, work_dir)
                transcription = speech_service.transcribe(str(audio_path))

                # Release Whisper from GPU before loading PANNs
                if settings.sequential_gpu and settings.device == "cuda":
                    speech_service.release_model()

                # Store transcript segments
                segment_records = await self._store_segments(
                    db, video, transcription.segments
                )

                # ── Phase 2b: Audio Intelligence (PANNs on GPU) ──────
                # Prefer 32 kHz audio for PANNs if available
                audio_hq = work_dir / "audio_hq.wav"
                panns_audio = str(audio_hq) if audio_hq.exists() else str(audio_path)
                audio_analysis = audio_intelligence_service.analyze(panns_audio)

                # Release PANNs from GPU before loading CLIP
                if settings.sequential_gpu and settings.device == "cuda":
                    audio_intelligence_service.release_model()

                # Store audio segments
                audio_records = await self._store_audio_segments(
                    db, video, audio_analysis
                )

                # ── Phase 2c: Alignment + Change Points (v4, CPU) ────
                # Compute audio-transcript alignment
                transcript_dicts = [
                    {
                        "start_time": s.start_time,
                        "end_time": s.end_time,
                        "text": s.text,
                        "confidence": s.confidence,
                    }
                    for s in transcription.segments
                ]
                audio_window_dicts = [
                    {
                        "start_time": w.start_time,
                        "end_time": w.end_time,
                        "speech_probability": w.speech_probability,
                        "music_probability": w.music_probability,
                        "dominant_source": (
                            w.source_tags[0].label if w.source_tags else "unknown"
                        ),
                        "event_tags": [
                            {"label": t.label, "confidence": t.confidence}
                            for t in (w.event_tags or [])
                        ],
                        "loudness_lufs": w.attributes.loudness_lufs,
                        "harmonic_ratio": w.attributes.harmonic_ratio,
                        "spectral_centroid": w.attributes.spectral_centroid,
                        "bpm": w.attributes.bpm,
                    }
                    for w in audio_analysis.windows
                ]

                duration = video.duration_seconds or 0
                alignment_result = alignment_service.compute_alignment(
                    transcript_dicts, audio_window_dicts, float(duration),
                )
                await self._store_alignments(db, video, alignment_result)

                # Detect acoustic change points
                cp_result = change_point_detector.detect(audio_window_dicts)
                await self._store_change_points(db, video, cp_result)

                # ── Phase 3: Frames + Vision + OCR (CLIP on GPU) ─────
                # Ensure CLIP is loaded for vision + embedding
                embedding_service.ensure_clip()
                vision_service.reset_cache()  # re-encode labels with fresh CLIP

                frames_data = self._extract_frames(str(media_path))
                ocr_results = self._run_ocr_pipeline(frames_data)
                visual_results = self._run_vision_pipeline(frames_data)

                frame_records = await self._store_frames(
                    db, video, frames_data, ocr_results, visual_results
                )

                # ── Phase 4: Embeddings + Indexing ───────────────────
                await self._index_embeddings(
                    video, segment_records, frame_records, frames_data
                )

                # Index audio embeddings into Qdrant
                await self._index_audio_embeddings(video, audio_records, audio_analysis)

                # Release CLIP from GPU when done
                if settings.sequential_gpu and settings.device == "cuda":
                    embedding_service.release_clip()
                    vision_service.reset_cache()

                # ── Phase 5: Mark complete ───────────────────────────
                video.status = VideoStatus.INDEXED
                video.model_version = embedding_service.current_model_version
                video.processing_error = None
                await db.commit()

                logger.info(
                    f"✓ Processed {video.platform.value}:{video.platform_id} — "
                    f"{len(segment_records)} segments, {len(frame_records)} frames, "
                    f"{len(audio_records)} audio windows, "
                    f"{len(cp_result.change_points)} change points, "
                    f"alignment={alignment_result.overall_quality}"
                )

            except Exception as e:
                logger.exception(f"✗ Failed to process video {video_id}: {e}")
                video.status = VideoStatus.FAILED
                video.processing_error = str(e)[:2000]
                await db.commit()

            finally:
                # Release ALL GPU models to prevent VRAM leaks on error
                if settings.sequential_gpu and settings.device == "cuda":
                    try:
                        speech_service.release_model()
                    except Exception:
                        pass
                    try:
                        audio_intelligence_service.release_model()
                    except Exception:
                        pass
                    try:
                        embedding_service.release_clip()
                        vision_service.reset_cache()
                    except Exception:
                        pass
                self._cleanup(work_dir)

    # ── Step 1: Download (Multi-Platform) ────────────────────────────────

    async def _download_video(
        self, video: Video, work_dir: Path
    ) -> Optional[Path]:
        """Download video using yt-dlp — supports YouTube, TikTok, Twitter, etc."""
        import yt_dlp

        platform = self.detect_platform(video.url)
        output_template = str(work_dir / "video.%(ext)s")

        # Platform-specific format selection
        format_str = self._get_download_format(platform)

        ydl_opts = {
            "format": format_str,
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "merge_output_format": "mkv",       # MKV handles all codecs
            "socket_timeout": 60,
            "retries": 10,                       # more retries for reliability
            "fragment_retries": 10,
            "file_access_retries": 5,
            "extractor_retries": 5,
            "age_limit": None,
            "geo_bypass": True,
            # Prefer Opus audio for maximum quality
            "postprocessors": [{
                "key": "FFmpegVideoConvertor",
                "preferedformat": "mkv",
            }],
            # Extract subtitles if available (bonus transcript source)
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "ar", "fr", "de", "es", "ja", "ko", "zh"],
            "subtitlesformat": "srt/vtt/best",
            # TikTok / Instagram specific
            "extractor_args": {
                "tiktok": {"api_hostname": "api22-normal-c-useast2a.tiktokv.com"},
            },
            # Metadata for maximum data collection
            "writeinfojson": True,
            "writethumbnail": True,
        }

        # Platform-specific tweaks
        if platform == Platform.TIKTOK:
            ydl_opts["format"] = settings.download_format_tiktok
            # TikTok videos are short — no merge needed
            ydl_opts.pop("merge_output_format", None)
        elif platform == Platform.TWITTER:
            ydl_opts["format"] = "best"
        elif platform == Platform.INSTAGRAM:
            ydl_opts["format"] = "best"

        try:
            logger.info(f"Downloading [{platform.value}]: {video.url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video.url])

            # Find the downloaded file
            for f in sorted(work_dir.iterdir(), key=lambda p: p.stat().st_size, reverse=True):
                if f.suffix in (".mp4", ".mkv", ".webm", ".mov", ".avi", ".flv", ".ts", ".m4v"):
                    logger.info(f"Downloaded: {f.name} ({f.stat().st_size / 1024 / 1024:.1f} MB)")
                    return f

        except Exception as e:
            logger.error(f"Download failed for {video.url}: {e}")
        return None

    def _get_download_format(self, platform: Platform) -> str:
        """Platform-specific format strings for MAXIMUM quality."""
        if platform == Platform.YOUTUBE:
            return settings.download_format_youtube
        elif platform == Platform.TIKTOK:
            return settings.download_format_tiktok
        elif platform in (Platform.TWITTER, Platform.INSTAGRAM, Platform.REDDIT):
            return "best"
        elif platform == Platform.TWITCH:
            return "best[height<=2160]"
        elif platform == Platform.VIMEO:
            return "bestvideo[height<=2160]+bestaudio/best[height<=2160]"
        elif platform == Platform.BILIBILI:
            return "bestvideo[height<=2160]+bestaudio/best"
        else:
            return settings.download_format_generic

    # ── Step 1b: Enrich from yt-dlp info + ffprobe ────────────────────

    async def _enrich_video_metadata(
        self, db: AsyncSession, video: Video, work_dir: Path, media_path: Path,
    ):
        """Read yt-dlp .info.json + ffprobe, populate Video fields and child records."""
        import json as _json

        # ── Read yt-dlp info.json ────────────────────────────────────
        info = {}
        for f in work_dir.iterdir():
            if f.suffix == ".json" and "info" in f.name:
                try:
                    info = _json.loads(f.read_text(errors="replace"))
                except Exception:
                    pass
                break

        if info:
            video.tags = info.get("tags") or None
            video.categories = info.get("categories") or None
            video.age_limit = info.get("age_limit")
            video.comment_count = info.get("comment_count")
            video.dislike_count = info.get("dislike_count")
            video.is_short = "shorts" in (info.get("webpage_url") or "")

            # Live status
            raw_live = info.get("live_status") or info.get("is_live")
            if raw_live == "is_live" or raw_live is True:
                video.live_status = LiveStatus.IS_LIVE
            elif raw_live == "was_live":
                video.live_status = LiveStatus.WAS_LIVE
            elif raw_live == "is_upcoming":
                video.live_status = LiveStatus.IS_UPCOMING
            elif raw_live == "post_live":
                video.live_status = LiveStatus.POST_LIVE
            else:
                video.live_status = LiveStatus.NONE

            # Captions availability
            subs = info.get("subtitles") or {}
            auto_subs = info.get("automatic_captions") or {}
            video.has_captions = bool(subs)
            video.caption_languages = list(subs.keys()) if subs else None
            video.has_auto_captions = bool(auto_subs)
            video.auto_caption_languages = list(auto_subs.keys())[:50] if auto_subs else None

            # Chapters → child records
            for ch in (info.get("chapters") or []):
                db.add(Chapter(
                    video_id=video.id,
                    title=ch.get("title", ""),
                    start_time=ch.get("start_time", 0),
                    end_time=ch.get("end_time"),
                ))

            # Subtitles → child records (from downloaded .srt/.vtt files)
            for lang, sub_list in {**subs, **auto_subs}.items():
                is_auto = lang in auto_subs and lang not in subs
                # Check if subtitle file was actually downloaded
                for ext in ("srt", "vtt", "ass"):
                    sub_file = None
                    for candidate in work_dir.glob(f"*.{lang}.{ext}"):
                        sub_file = candidate
                        break
                    if sub_file and sub_file.exists():
                        try:
                            content = sub_file.read_text(errors="replace")[:500_000]
                            cue_count = content.count("\n\n")
                        except Exception:
                            content, cue_count = None, 0
                        db.add(Subtitle(
                            video_id=video.id,
                            language=lang,
                            is_auto_generated=is_auto,
                            format=ext,
                            content=content,
                            cue_count=max(cue_count, 1),
                        ))
                        break

            # Overflow metadata (anything yt-dlp returns beyond our columns)
            video.metadata_json = {
                k: v for k, v in info.items()
                if k in (
                    "webpage_url", "extractor", "format_id", "format_note",
                    "fps", "width", "height", "vcodec", "acodec", "abr", "vbr",
                    "tbr", "asr", "filesize_approx", "stretched_ratio",
                    "availability", "original_url", "channel_follower_count",
                    "uploader_url", "license", "series", "season_number",
                    "episode_number", "track", "artist", "album", "release_year",
                )
                and v is not None
            } or None

        # ── Stream info via ffprobe ──────────────────────────────────
        try:
            import json as _j
            probe_out = subprocess.run(
                [
                    "ffprobe", "-v", "quiet", "-print_format", "json",
                    "-show_streams", str(media_path),
                ],
                capture_output=True, text=True, timeout=30,
            )
            probe = _j.loads(probe_out.stdout) if probe_out.returncode == 0 else {}
            for s in probe.get("streams", []):
                codec_type = s.get("codec_type", "")
                if codec_type not in ("video", "audio"):
                    continue
                rec = StreamInfo(
                    video_id=video.id,
                    stream_type=codec_type,
                    codec=s.get("codec_name"),
                    bitrate=int(s["bit_rate"]) if s.get("bit_rate") else None,
                    container_format=probe.get("format", {}).get("format_name"),
                )
                if codec_type == "video":
                    w, h = s.get("width"), s.get("height")
                    rec.resolution = f"{w}x{h}" if w and h else None
                    fps_str = s.get("r_frame_rate", "")
                    if "/" in fps_str:
                        num, den = fps_str.split("/")
                        rec.fps = round(int(num) / max(int(den), 1), 2) if num.isdigit() else None
                elif codec_type == "audio":
                    rec.sample_rate = int(s["sample_rate"]) if s.get("sample_rate") else None
                    rec.channels = s.get("channels")
                db.add(rec)
        except Exception as e:
            logger.debug(f"ffprobe skipped: {e}")

        await db.flush()

    # ── Step 2: Audio Extraction ─────────────────────────────────────────

    @staticmethod
    def _extract_audio(video_path: Path, work_dir: Path) -> Path:
        """Extract audio to WAV 16 kHz mono for Whisper.
        Also extracts 32 kHz version for PANNs if needed.
        """
        audio_path = work_dir / "audio.wav"
        cmd = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd, check=True, timeout=600)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg audio extraction failed: {e}")
            raise RuntimeError(f"Audio extraction failed: {e}") from e
        except subprocess.TimeoutExpired:
            raise RuntimeError("Audio extraction timed out (10 min limit)")

        # Also extract 32 kHz version for PANNs (higher fidelity)
        audio_hq = work_dir / "audio_hq.wav"
        cmd_hq = [
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "32000", "-ac", "1",
            str(audio_hq),
            "-y", "-loglevel", "error",
        ]
        try:
            subprocess.run(cmd_hq, check=True, timeout=600)
        except Exception:
            logger.warning("Failed to extract 32 kHz audio — PANNs will use 16 kHz")

        return audio_path

    # ── Step 3: Store Segments ───────────────────────────────────────────

    async def _store_segments(
        self, db: AsyncSession, video: Video, segments
    ) -> List[Segment]:
        records = []
        for i, seg in enumerate(segments):
            if i >= settings.max_segments_per_video:
                logger.warning(f"Segment limit reached ({settings.max_segments_per_video})")
                break
            record = Segment(
                video_id=video.id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                text=seg.text,
                language=seg.language,
                confidence=seg.confidence,
                speaker_label=seg.speaker_label,
                model_version=f"whisper-{settings.whisper_model}",
            )
            db.add(record)
            records.append(record)
        await db.flush()
        return records

    # ── Step 4: Frame Extraction ─────────────────────────────────────────

    def _extract_frames(
        self, video_path: str
    ) -> List[Tuple[float, np.ndarray]]:
        """Sample frames at dense intervals, detect scene changes, filter blur."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = max(fps, 1.0)  # Guard against zero/negative FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        interval = settings.frame_sample_interval
        frame_interval_count = max(1, int(fps * interval))

        frames: List[Tuple[float, np.ndarray]] = []
        prev_frame = None
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Sample at interval using frame count (numerically stable)
            if frame_idx % frame_interval_count != 0:
                frame_idx += 1
                continue

            timestamp = frame_idx / max(fps, 1.0)

            # Scene change detection — always include scene changes
            is_scene_change = False
            if prev_frame is not None:
                is_scene_change = vision_service.detect_scene_change(prev_frame, frame)

            # Blur detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

            if blur_score < settings.blur_threshold and not is_scene_change:
                prev_frame = frame
                frame_idx += 1
                continue  # Skip very blurry frames unless scene change

            frames.append((round(timestamp, 3), frame))
            prev_frame = frame
            frame_idx += 1

            if len(frames) >= settings.max_frames_per_video:
                logger.warning(f"Frame limit reached ({settings.max_frames_per_video})")
                break

        cap.release()
        logger.info(f"Extracted {len(frames)} frames from {duration:.0f}s video (interval={interval}s)")
        return frames

    # ── Step 5: OCR Pipeline ─────────────────────────────────────────────

    def _run_ocr_pipeline(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> List[MergedOCRResult]:
        """Run dual-engine OCR with temporal smoothing on all frames."""
        raw_results = []
        for i, (ts, frame) in enumerate(frames):
            result = ocr_service.extract_ocr(frame)
            raw_results.append(result)
            if (i + 1) % 100 == 0:
                logger.debug(f"OCR progress: {i + 1}/{len(frames)} frames")

        smoothed = ocr_service.temporal_smooth(raw_results)
        ocr_count = sum(1 for r in smoothed if r.text.strip())
        logger.info(f"OCR complete: {ocr_count}/{len(frames)} frames had text")
        return smoothed

    # ── Step 6: Vision Pipeline ──────────────────────────────────────────

    def _run_vision_pipeline(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> List:
        """Run CLIP-based visual classification on frames."""
        pil_images = [
            Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            for _, f in frames
        ]

        results = []
        batch_size = 8  # conservative for ViT-H-14 on 12 GB
        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i : i + batch_size]
            batch_results = vision_service.classify_frames_batch(batch, top_k=7)
            results.extend(batch_results)
            if (i + batch_size) % 50 == 0:
                logger.debug(f"Vision progress: {min(i + batch_size, len(pil_images))}/{len(pil_images)}")

        return results

    # ── Step 7: Store Frames ─────────────────────────────────────────────

    async def _store_frames(
        self,
        db: AsyncSession,
        video: Video,
        frames_data: List[Tuple[float, np.ndarray]],
        ocr_results: List[MergedOCRResult],
        visual_results: List,
    ) -> List[Frame]:
        records = []
        for i, (ts, _) in enumerate(frames_data):
            ocr = ocr_results[i] if i < len(ocr_results) else None
            vis = visual_results[i] if i < len(visual_results) else None

            visual_tags = None
            scene_label = None
            is_scene_change = False
            if vis:
                visual_tags = [
                    {"label": t.label, "confidence": t.confidence, "category": t.category}
                    for t in vis.tags[:15]  # store more tags for richer search
                ]
                scene_label = vis.scene_label
                is_scene_change = getattr(vis, "is_scene_change", False)

            record = Frame(
                video_id=video.id,
                timestamp=ts,
                visual_tags=visual_tags,
                ocr_text=ocr.text if ocr and ocr.text else None,
                ocr_confidence=ocr.confidence if ocr else None,
                scene_label=scene_label,
                is_scene_change=is_scene_change,
                model_version=embedding_service.current_model_version,
            )
            db.add(record)
            records.append(record)

        await db.flush()
        return records

    # ── Step 8: Embedding Indexing ───────────────────────────────────────

    async def _index_embeddings(
        self,
        video: Video,
        segments: List[Segment],
        frames: List[Frame],
        frames_data: List[Tuple[float, np.ndarray]],
    ):
        """Generate embeddings and upsert into Qdrant."""
        vid_str = str(video.id)

        # Delete old vectors first (for re-indexing)
        vector_store.delete_by_video(vid_str)

        # ── Text embeddings ──────────────────────────────────────────
        if segments:
            texts = [s.text for s in segments]
            embeddings = embedding_service.embed_texts(texts)
            payloads = [
                {
                    "video_id": vid_str,
                    "segment_id": str(s.id),
                    "timestamp": s.start_time,
                    "end_time": s.end_time,
                    "text": s.text[:1000],  # more text for keyword matching
                    "confidence": s.confidence,
                    "language": s.language or "en",
                    "channel_id": str(video.channel_id) if video.channel_id else "",
                    "platform": video.platform.value if video.platform else "other",
                    "model_version": embedding_service.current_model_version,
                    "ocr_text": "",  # enriched below
                }
                for s in segments
            ]

            # Enrich segments with nearby OCR text
            for i, seg in enumerate(segments):
                nearby_ocr = [
                    f.ocr_text
                    for f in frames
                    if f.ocr_text and abs(f.timestamp - seg.start_time) < 20
                ]
                if nearby_ocr:
                    payloads[i]["ocr_text"] = " ".join(set(nearby_ocr[:5]))

            ids = [str(s.id) for s in segments]
            vector_store.upsert_text_embeddings(list(embeddings), payloads, ids)
            logger.info(f"Indexed {len(segments)} text embeddings for {vid_str}")

        # ── Visual embeddings ────────────────────────────────────────
        if frames_data and frames:
            pil_images = [
                Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
                for _, f in frames_data
            ]

            batch_size = 16
            all_embeddings = []
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i : i + batch_size]
                batch_embs = embedding_service.embed_images(batch)
                all_embeddings.append(batch_embs)

            if all_embeddings:
                vis_embeddings = np.concatenate(all_embeddings, axis=0)
                count = min(len(vis_embeddings), len(frames_data), len(frames))

                vis_payloads = [
                    {
                        "video_id": vid_str,
                        "frame_id": str(frames[i].id),
                        "timestamp": frames_data[i][0],
                        "visual_tags": frames[i].visual_tags or [],
                        "ocr_text": frames[i].ocr_text or "",
                        "scene_label": frames[i].scene_label or "",
                        "channel_id": str(video.channel_id) if video.channel_id else "",
                        "platform": video.platform.value if video.platform else "other",
                        "model_version": embedding_service.current_model_version,
                    }
                    for i in range(count)
                ]

                vis_ids = [str(frames[i].id) for i in range(count)]
                vector_store.upsert_visual_embeddings(
                    list(vis_embeddings[:count]), vis_payloads, vis_ids,
                )
                logger.info(f"Indexed {count} visual embeddings for {vid_str}")

    # ── Step 7b: Store Audio Segments ───────────────────────────────────

    async def _store_audio_segments(
        self,
        db: AsyncSession,
        video: Video,
        analysis,
    ) -> List[AudioSegment]:
        """Store audio intelligence results to PostgreSQL with calibration."""
        records = []
        for window in analysis.windows:
            dominant = "unknown"
            if window.source_tags:
                dominant = window.source_tags[0].label

            # Calibrate manipulation scores (v4)
            calibrated = {}
            manip = window.manipulation
            try:
                for field_name in (
                    "speed_anomaly", "pitch_shift", "compression",
                    "reverb_echo", "robotic_autotune", "time_stretch",
                ):
                    raw = getattr(manip, field_name, 0)
                    cal = calibration_service.calibrate(f"audio.{field_name}", raw)
                    calibrated[field_name] = cal.to_dict()
            except Exception as e:
                logger.debug(f"Calibration fallback: {e}")

            record = AudioSegment(
                video_id=video.id,
                start_time=window.start_time,
                end_time=window.end_time,
                music_probability=round(window.music_probability, 3),
                speech_probability=round(window.speech_probability, 3),
                dominant_source=dominant,
                source_tags=[
                    {"label": t.label, "confidence": t.confidence, "category": t.category}
                    for t in window.source_tags[:5]
                ] if window.source_tags else None,
                event_tags=[
                    {"label": t.label, "confidence": t.confidence, "category": t.category}
                    for t in window.event_tags[:10]
                ] if window.event_tags else None,
                manipulation_scores=window.manipulation.to_dict(),
                overall_manipulation=window.manipulation.overall_manipulation,
                bpm=round(window.attributes.bpm, 1) if window.attributes.bpm else None,
                musical_key=window.attributes.key,
                loudness_lufs=round(window.attributes.loudness_lufs, 1),
                dynamic_range_db=round(window.attributes.dynamic_range_db, 1),
                spectral_centroid=round(window.attributes.spectral_centroid, 1),
                spectral_rolloff=round(window.attributes.spectral_rolloff, 1),
                harmonic_ratio=round(window.attributes.harmonic_ratio, 3),
                zero_crossing_rate=round(window.attributes.zero_crossing_rate, 5),
                calibrated_manipulation=calibrated if calibrated else None,
                calibration_version=calibration_service.version,
                model_version="panns-cnn14",
            )
            db.add(record)
            records.append(record)

        await db.flush()
        logger.info(f"Stored {len(records)} audio segments for {video.id}")
        return records

    # ── Step 7c: Store Alignments (v4) ───────────────────────────────────

    async def _store_alignments(
        self,
        db: AsyncSession,
        video: Video,
        alignment_result,
    ):
        """Store audio-transcript alignment results."""
        for seg in alignment_result.segments:
            record = TranscriptAlignment(
                video_id=video.id,
                start_time=seg.start_time,
                end_time=seg.end_time,
                alignment_score=round(seg.alignment_score, 3),
                quality_level=seg.quality_level,
                signals_json=[
                    {"name": s.name, "score": round(s.score, 3), "detail": s.detail}
                    for s in seg.signals
                ],
                warnings_json=seg.warnings if seg.warnings else None,
            )
            db.add(record)

        await db.flush()
        logger.info(
            f"Stored {len(alignment_result.segments)} alignment records for {video.id} "
            f"(overall={alignment_result.overall_score:.2f}, quality={alignment_result.overall_quality})"
        )

    # ── Step 7d: Store Change Points (v4) ────────────────────────────────

    async def _store_change_points(
        self,
        db: AsyncSession,
        video: Video,
        cp_result,
    ):
        """Store acoustic change points."""
        for cp in cp_result.change_points:
            record = AcousticChangePoint(
                video_id=video.id,
                timestamp=round(cp.timestamp, 2),
                magnitude=round(cp.magnitude, 3),
                transition_type=cp.transition_type.value,
                detail=cp.detail[:500] if cp.detail else None,
                from_state=cp.from_state[:64] if cp.from_state else None,
                to_state=cp.to_state[:64] if cp.to_state else None,
                spectral_distance=round(cp.spectral_distance, 3),
                energy_ratio=round(cp.energy_ratio, 3),
            )
            db.add(record)

        await db.flush()
        logger.info(
            f"Stored {len(cp_result.change_points)} change points for {video.id} "
            f"({cp_result.num_scenes} acoustic scenes)"
        )

    # ── Step 8b: Audio Embedding Indexing ─────────────────────────────

    async def _index_audio_embeddings(
        self,
        video: Video,
        records: List[AudioSegment],
        analysis,
    ):
        """Index audio PANNs embeddings into Qdrant for acoustic search."""
        vid_str = str(video.id)

        embeddings = []
        payloads = []
        ids = []

        for i, window in enumerate(analysis.windows):
            if window.embedding is None or i >= len(records):
                continue

            rec = records[i]
            embeddings.append(window.embedding)
            ids.append(str(rec.id))
            payloads.append({
                "video_id": vid_str,
                "audio_segment_id": str(rec.id),
                "start_time": window.start_time,
                "end_time": window.end_time,
                "dominant_source": rec.dominant_source or "unknown",
                "music_probability": window.music_probability,
                "speech_probability": window.speech_probability,
                "event_labels": [t.label for t in window.event_tags[:5]],
                "bpm": rec.bpm,
                "musical_key": rec.musical_key,
                "loudness_lufs": rec.loudness_lufs,
                "harmonic_ratio": rec.harmonic_ratio,
                "overall_manipulation": rec.overall_manipulation,
                "channel_id": str(video.channel_id) if video.channel_id else "",
                "platform": video.platform.value if video.platform else "other",
                "model_version": "panns-cnn14",
            })

        if embeddings:
            vector_store.upsert_audio_embeddings(embeddings, payloads, ids)
            logger.info(f"Indexed {len(embeddings)} audio embeddings for {vid_str}")

    # ── Cleanup ──────────────────────────────────────────────────────────

    @staticmethod
    def _cleanup(work_dir: Path):
        """Remove temporary files."""
        import shutil
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")


media_service = MediaProcessingService()
