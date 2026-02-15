# Nexum v4.2 — Deep Signals + Recrawl Intelligence

## What Changed

### 24 New Signal Models (ORM)

**Temporal-Visual (4)**
- `CursorHeatmap` — mouse movement tracking in screen recordings, dwell time, velocity, idle ratio
- `TextEditEvent` — typed/deleted/corrected text in coding tutorials, syntax vs logic error fixes
- `FaceCameraRatio` — face visibility over time per 30s window, PiP detection, face area ratio
- `BackgroundChangeEvent` — studio/background changes between uploads per channel, embedding distance

**Audio Micro-Signals (4)**
- `BreathingPattern` — breath type (deep inhale/gasp/sigh), intensity, pre/post speech tempo
- `AmbientAudioBleed` — keyboard clicks, mouse bleed, notifications; proves live coding vs playback
- `RoomAcousticFingerprint` — RT60, clarity C50, spectral fingerprint (32-dim), room cluster ID
- `LaughterEvent` — authenticity score via spectral decay, pitch variability, trigger context

**Behavioral Metadata (4)**
- `UploadPattern` — upload hour/day, days since last upload, drift from channel mean, anomaly flag
- `TitleEditHistory` — title/thumbnail/description/tag edits with hours-after-upload timestamp
- `ThumbnailAnalysis` — dominant colors, face/text detection, clickbait score, palette distance
- `DescriptionLinkHealth` — URL status checks, affiliate detection, self-reference flag, link rot

**Comment Archaeology (3)**
- `PinnedCommentHistory` — pinned comment changes over time, unpin detection
- `CommentSentimentDrift` — sentiment at hour_1/day_1/day_7/day_30/day_90, language distribution
- `DeletedCommentShadow` — orphaned reply chains, last known text/author, probable deletion reason

**Cross-Video Forensics (3)**
- `BRollReuse` — perceptual hash matching across videos, stock footage detection
- `SponsorSegment` — sponsor/self-promo/merch segments with position ratio and confidence
- `IntroOutroEvolution` — intro/outro presence tracking, fingerprint hashing, silence-before-ask timing

**Linguistic Signals (1)**
- `LinguisticProfile` — code-switching pairs, filler word clusters at knowledge boundaries, hedging rate by topic, Zipf deviation, vocabulary richness, speaking rate WPM

**Graph-Relational (4)**
- `CommenterMigration` — audience movement between channels with overlap period
- `EntityPropagation` — first-mention attribution and propagation chain with adoption lag
- `ParasocialIndex` — creator reply ratio per commenter, engagement tier (superfan/regular/casual)
- `TopicGestation` — time from first comment mention to creator video, audience-requested flag

**Recrawl (1)**
- `RecrawlEvent` — trigger type, field deltas, comment add/delete counts, description/title change flags

### Universal `updated_at`

Every model now has `updated_at` with `onupdate=func.now()`. Total: 40 of 48 models (remaining 8 are junction/helper tables where it's unnecessary). All demo-generated nodes include `updated_at` for staleness detection.

### New API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/demo/signals/types` | List all 24 signal types with counts |
| GET | `/demo/signals/stats` | Aggregate stats by category |
| GET | `/demo/signals/{type}` | Paginated signal records, filterable by video_id/channel_id |
| POST | `/demo/recrawl/{video_id}` | Simulate recrawl — returns delta of what changed |
| GET | `/demo/stale` | Find nodes needing recrawl, sorted by oldest updated_at |

### Demo Data Volume

| Category | Records |
|----------|---------|
| breathing_patterns | 2,398 |
| face_camera_ratios | 2,023 |
| cursor_heatmaps | 1,575 |
| text_edit_events | 752 |
| sentiment_drifts | 750 |
| ambient_bleeds | 620 |
| link_health | 408 |
| laughter_events | 376 |
| room_fingerprints | 240 |
| upload_patterns | 240 |
| thumbnail_analyses | 240 |
| intro_outro_evolutions | 240 |
| linguistic_profiles | 240 |
| sponsor_segments | 151 |
| deleted_shadows | 132 |
| parasocial_indices | 120 |
| commenter_migrations | 100 |
| recrawl_events | 80 |
| broll_reuses | 60 |
| topic_gestations | 60 |
| pinned_comment_history | 56 |
| title_edit_history | 51 |
| entity_propagations | 40 |
| background_changes | 34 |
| **TOTAL** | **~11,000** |

### Files Modified/Created

| File | Action | Lines |
|------|--------|-------|
| `backend/app/models/models.py` | Modified | ~1,100 |
| `backend/app/services/demo/signal_generators.py` | Created | ~580 |
| `backend/app/services/demo/demo_data_generator.py` | Modified | ~915 |
| `backend/app/api/routes/demo.py` | Rewritten | ~290 |
| `backend/alembic/versions/v4_2_signals.py` | Existing | ~470 |
