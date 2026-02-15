"""
Nexum Signal Generators — v4.2 Deep Analytics.

Generates realistic synthetic data for every signal type:
  - Temporal-Visual: cursor heatmaps, text edits, face ratios, background changes
  - Audio Micro: breathing, ambient bleed, room fingerprints, laughter
  - Behavioral: upload patterns, title edits, thumbnails, link health
  - Comment Archaeology: pinned history, sentiment drift, deleted shadows
  - Cross-Video Forensics: b-roll reuse, sponsor segments, intro/outro
  - Linguistic: code-switching, fillers, hedging, vocabulary
  - Graph-Relational: migrations, entity propagation, parasocial, gestation
  - Recrawl: recrawl event tracking

Every generated node includes `updated_at` for staleness tracking.
"""
from __future__ import annotations

import hashlib
import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


def _uid(prefix: str, i: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"nexum.signal.{prefix}.{i}"))


def _ts(days_back: int = 400) -> str:
    dt = datetime.now(timezone.utc) - timedelta(
        days=random.randint(0, days_back),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return dt.isoformat()


def _updated_at(created_at_iso: str) -> str:
    """Generate an updated_at that's 0-30 days after created_at."""
    try:
        base = datetime.fromisoformat(created_at_iso)
    except (ValueError, TypeError):
        base = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 200))
    delta = timedelta(days=random.randint(0, 30), hours=random.randint(0, 12))
    return (base + delta).isoformat()


def inject_updated_at(nodes: List[Dict]) -> List[Dict]:
    """Add updated_at to every node that has created_at but no updated_at."""
    now = datetime.now(timezone.utc).isoformat()
    for n in nodes:
        data = n.get("data", n)
        if "updated_at" not in data:
            ca = data.get("created_at") or data.get("first_seen_at") or data.get("detected_at")
            data["updated_at"] = _updated_at(ca) if ca else now
    return nodes


class SignalGenerator:
    """Generates deep-analytics signals attached to existing demo graph nodes."""

    def __init__(self, videos, channels, comments, authors, entities, topics):
        self._videos = videos
        self._channels = channels
        self._comments = comments
        self._authors = authors
        self._entities = entities
        self._topics = topics

        # All generated signals
        self.recrawl_events: List[Dict] = []
        self.cursor_heatmaps: List[Dict] = []
        self.text_edit_events: List[Dict] = []
        self.face_camera_ratios: List[Dict] = []
        self.background_changes: List[Dict] = []
        self.breathing_patterns: List[Dict] = []
        self.ambient_bleeds: List[Dict] = []
        self.room_fingerprints: List[Dict] = []
        self.laughter_events: List[Dict] = []
        self.upload_patterns: List[Dict] = []
        self.title_edit_history: List[Dict] = []
        self.thumbnail_analyses: List[Dict] = []
        self.link_health: List[Dict] = []
        self.pinned_comment_history: List[Dict] = []
        self.sentiment_drifts: List[Dict] = []
        self.deleted_shadows: List[Dict] = []
        self.broll_reuses: List[Dict] = []
        self.sponsor_segments: List[Dict] = []
        self.intro_outro_evolutions: List[Dict] = []
        self.linguistic_profiles: List[Dict] = []
        self.commenter_migrations: List[Dict] = []
        self.entity_propagations: List[Dict] = []
        self.parasocial_indices: List[Dict] = []
        self.topic_gestations: List[Dict] = []

    def generate_all(self):
        self._gen_recrawl_events()
        self._gen_cursor_heatmaps()
        self._gen_text_edit_events()
        self._gen_face_camera_ratios()
        self._gen_background_changes()
        self._gen_breathing_patterns()
        self._gen_ambient_bleeds()
        self._gen_room_fingerprints()
        self._gen_laughter_events()
        self._gen_upload_patterns()
        self._gen_title_edit_history()
        self._gen_thumbnail_analyses()
        self._gen_link_health()
        self._gen_pinned_comment_history()
        self._gen_sentiment_drifts()
        self._gen_deleted_shadows()
        self._gen_broll_reuses()
        self._gen_sponsor_segments()
        self._gen_intro_outro_evolutions()
        self._gen_linguistic_profiles()
        self._gen_commenter_migrations()
        self._gen_entity_propagations()
        self._gen_parasocial_indices()
        self._gen_topic_gestations()

    def stats(self) -> Dict[str, int]:
        return {
            "recrawl_events": len(self.recrawl_events),
            "cursor_heatmaps": len(self.cursor_heatmaps),
            "text_edit_events": len(self.text_edit_events),
            "face_camera_ratios": len(self.face_camera_ratios),
            "background_changes": len(self.background_changes),
            "breathing_patterns": len(self.breathing_patterns),
            "ambient_bleeds": len(self.ambient_bleeds),
            "room_fingerprints": len(self.room_fingerprints),
            "laughter_events": len(self.laughter_events),
            "upload_patterns": len(self.upload_patterns),
            "title_edit_history": len(self.title_edit_history),
            "thumbnail_analyses": len(self.thumbnail_analyses),
            "link_health": len(self.link_health),
            "pinned_comment_history": len(self.pinned_comment_history),
            "sentiment_drifts": len(self.sentiment_drifts),
            "deleted_shadows": len(self.deleted_shadows),
            "broll_reuses": len(self.broll_reuses),
            "sponsor_segments": len(self.sponsor_segments),
            "intro_outro_evolutions": len(self.intro_outro_evolutions),
            "linguistic_profiles": len(self.linguistic_profiles),
            "commenter_migrations": len(self.commenter_migrations),
            "entity_propagations": len(self.entity_propagations),
            "parasocial_indices": len(self.parasocial_indices),
            "topic_gestations": len(self.topic_gestations),
        }

    # ── Recrawl Events ───────────────────────────────────────────────

    def _gen_recrawl_events(self):
        for i, vid in enumerate(random.sample(self._videos, k=min(80, len(self._videos)))):
            created = _ts(200)
            triggers = ["scheduled", "scheduled", "scheduled", "drift_detected", "manual", "input_duplicate"]
            fields = random.sample(["view_count", "comment_count", "like_count", "description", "title", "tags", "thumbnail_url"], k=random.randint(1, 4))
            delta = {}
            if "view_count" in fields:
                old = vid.get("view_count", 1000)
                delta["view_count"] = [old, old + random.randint(100, 50000)]
            if "comment_count" in fields:
                old = vid.get("comment_count", 10)
                delta["comment_count"] = [old, old + random.randint(5, 500)]

            self.recrawl_events.append({
                "id": _uid("recrawl", i),
                "video_id": vid["id"],
                "trigger": random.choice(triggers),
                "fields_changed": fields,
                "delta_json": delta,
                "comments_added": random.randint(0, 200),
                "comments_deleted": random.randint(0, 15),
                "description_changed": "description" in fields,
                "title_changed": "title" in fields,
                "duration_ms": random.randint(800, 15000),
                "created_at": created,
                "updated_at": _updated_at(created),
            })

    # ── Temporal-Visual ──────────────────────────────────────────────

    def _gen_cursor_heatmaps(self):
        tech_vids = [v for v in self._videos if v.get("category") in ("tech",)]
        for i, vid in enumerate(tech_vids[:60]):
            dur = vid.get("duration_seconds", 600)
            window = random.choice([10, 15, 30])
            for j in range(min(dur // max(window, 1), 30)):
                start = j * window
                created = vid.get("created_at", _ts())
                cells = [{"x": random.randint(0, 15), "y": random.randint(0, 9),
                          "dwell_ms": random.randint(50, 3000), "clicks": random.randint(0, 5)}
                         for _ in range(random.randint(3, 12))]
                self.cursor_heatmaps.append({
                    "id": _uid(f"cursor_{vid['id']}", j),
                    "video_id": vid["id"],
                    "start_time": float(start),
                    "end_time": float(min(start + window, dur)),
                    "grid_cells": cells,
                    "hotspot_x": round(random.uniform(0.2, 0.8), 3),
                    "hotspot_y": round(random.uniform(0.3, 0.7), 3),
                    "total_distance_px": round(random.uniform(200, 8000), 1),
                    "avg_velocity": round(random.uniform(20, 600), 1),
                    "idle_ratio": round(random.uniform(0.1, 0.7), 3),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_text_edit_events(self):
        tech_vids = [v for v in self._videos if v.get("category") in ("tech",)]
        actions = ["typed", "deleted", "corrected", "pasted", "undone"]
        langs = ["python", "rust", "typescript", "go", "bash", "javascript", "c++"]
        for i, vid in enumerate(tech_vids[:50]):
            n_events = random.randint(3, 25)
            for j in range(n_events):
                created = vid.get("created_at", _ts())
                self.text_edit_events.append({
                    "id": _uid(f"textedit_{vid['id']}", j),
                    "video_id": vid["id"],
                    "timestamp": round(random.uniform(10, vid.get("duration_seconds", 600) - 10), 2),
                    "action": random.choice(actions),
                    "text_before": f"fn main() {{ /* error */ }}" if random.random() < 0.3 else None,
                    "text_after": f"fn main() {{ println!(\"fixed\"); }}" if random.random() < 0.3 else None,
                    "keystroke_count": random.randint(5, 200),
                    "time_to_correction_ms": random.randint(500, 30000) if random.random() < 0.4 else None,
                    "is_syntax_error_fix": random.random() < 0.25,
                    "is_logic_error_fix": random.random() < 0.1,
                    "programming_language": random.choice(langs),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_face_camera_ratios(self):
        for i, vid in enumerate(self._videos[:120]):
            dur = vid.get("duration_seconds", 300)
            window = 30
            for j in range(min(dur // max(window, 1), 20)):
                start = j * window
                created = vid.get("created_at", _ts())
                self.face_camera_ratios.append({
                    "id": _uid(f"face_{vid['id']}", j),
                    "video_id": vid["id"],
                    "start_time": float(start),
                    "end_time": float(min(start + window, dur)),
                    "face_visible_ratio": round(random.betavariate(2, 3), 3),
                    "face_count": random.choices([0, 1, 1, 1, 2], weights=[15, 50, 50, 50, 10])[0],
                    "face_area_ratio": round(random.uniform(0.02, 0.45), 3),
                    "is_facecam_overlay": random.random() < 0.15,
                    "background_changed": random.random() < 0.05,
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_background_changes(self):
        for ch in self._channels:
            ch_vids = sorted(
                [v for v in self._videos if v.get("channel_id") == ch["id"]],
                key=lambda x: x.get("uploaded_at") or x.get("created_at", "")
            )
            for j in range(len(ch_vids) - 1):
                if random.random() < 0.15:  # 15% chance of background change between uploads
                    created = ch_vids[j + 1].get("created_at", _ts())
                    self.background_changes.append({
                        "id": _uid(f"bgchange_{ch['id']}", j),
                        "channel_id": ch["id"],
                        "video_id_before": ch_vids[j]["id"],
                        "video_id_after": ch_vids[j + 1]["id"],
                        "similarity_score": round(random.uniform(0.1, 0.6), 3),
                        "change_type": random.choice(["studio_upgrade", "moved", "lighting_change", "equipment_change"]),
                        "embedding_distance": round(random.uniform(0.3, 1.5), 3),
                        "detected_at": created,
                        "updated_at": _updated_at(created),
                    })

    # ── Audio Micro-Signals ──────────────────────────────────────────

    def _gen_breathing_patterns(self):
        types = ["deep_inhale", "quick_gasp", "sigh", "held", "panting"]
        contexts = ["pre_statement", "post_question", "filler", "topic_transition", "emphasis"]
        for i, vid in enumerate(self._videos[:100]):
            n = random.randint(5, 40)
            for j in range(n):
                created = vid.get("created_at", _ts())
                self.breathing_patterns.append({
                    "id": _uid(f"breath_{vid['id']}", j),
                    "video_id": vid["id"],
                    "timestamp": round(random.uniform(2, vid.get("duration_seconds", 300) - 2), 2),
                    "breath_type": random.choice(types),
                    "duration_ms": random.randint(200, 2500),
                    "intensity": round(random.uniform(0.1, 0.9), 3),
                    "context": random.choice(contexts),
                    "preceding_speech_tempo": round(random.uniform(1.5, 5.0), 2),
                    "following_speech_tempo": round(random.uniform(1.5, 5.0), 2),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_ambient_bleeds(self):
        bleed_types = ["keyboard", "mouse_click", "chair_creak", "fan_hum", "notification", "door", "traffic"]
        for i, vid in enumerate(self._videos[:80]):
            n = random.randint(1, 15)
            for j in range(n):
                bt = random.choice(bleed_types)
                created = vid.get("created_at", _ts())
                self.ambient_bleeds.append({
                    "id": _uid(f"ambient_{vid['id']}", j),
                    "video_id": vid["id"],
                    "start_time": round(random.uniform(0, vid.get("duration_seconds", 300) - 5), 2),
                    "end_time": round(random.uniform(1, 8), 2),
                    "bleed_type": bt,
                    "event_count": random.randint(1, 50) if bt == "keyboard" else random.randint(1, 5),
                    "confidence": round(random.uniform(0.3, 0.95), 3),
                    "implies_live_input": bt in ("keyboard", "mouse_click"),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_room_fingerprints(self):
        cluster_counter = 0
        for ch in self._channels:
            ch_vids = [v for v in self._videos if v.get("channel_id") == ch["id"]]
            if not ch_vids:
                continue
            n_rooms = random.randint(1, 3)
            clusters = [f"room_{ch['id'][:8]}_{r}" for r in range(n_rooms)]
            for j, vid in enumerate(ch_vids):
                created = vid.get("created_at", _ts())
                cluster = random.choice(clusters)
                self.room_fingerprints.append({
                    "id": _uid(f"roomfp_{vid['id']}", 0),
                    "video_id": vid["id"],
                    "rt60_ms": round(random.uniform(100, 800), 1),
                    "early_decay_time_ms": round(random.uniform(50, 400), 1),
                    "clarity_c50": round(random.uniform(-5, 15), 2),
                    "spectral_fingerprint": [round(random.gauss(0, 1), 4) for _ in range(32)],
                    "room_size_estimate": random.choice(["small", "medium", "large", "outdoor"]),
                    "noise_floor_db": round(random.uniform(-60, -25), 1),
                    "matches_previous": j > 0 and random.random() < 0.7,
                    "cluster_id": cluster,
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    def _gen_laughter_events(self):
        for i, vid in enumerate(self._videos[:90]):
            n = random.randint(0, 8)
            for j in range(n):
                created = vid.get("created_at", _ts())
                self.laughter_events.append({
                    "id": _uid(f"laugh_{vid['id']}", j),
                    "video_id": vid["id"],
                    "timestamp": round(random.uniform(5, vid.get("duration_seconds", 300) - 5), 2),
                    "duration_ms": random.randint(300, 4000),
                    "authenticity_score": round(random.betavariate(3, 2), 3),
                    "spectral_decay_rate": round(random.uniform(0.5, 5.0), 3),
                    "pitch_variability": round(random.uniform(0.1, 2.0), 3),
                    "is_audience": random.random() < 0.1,
                    "trigger_context": random.choice([
                        "self-deprecating joke about code",
                        "unexpected demo result",
                        "audience question callback",
                        "ironic understatement",
                        None,
                    ]),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    # ── Behavioral Metadata ──────────────────────────────────────────

    def _gen_upload_patterns(self):
        for vid in self._videos:
            ch = next((c for c in self._channels if c["id"] == vid.get("channel_id")), None)
            if not ch:
                continue
            hour = random.choices(list(range(24)), weights=[
                1, 1, 1, 1, 1, 2, 3, 5, 8, 10, 12, 10,
                9, 8, 10, 12, 11, 9, 7, 5, 3, 2, 1, 1
            ])[0]
            created = vid.get("created_at", _ts())
            self.upload_patterns.append({
                "id": _uid(f"upat_{vid['id']}", 0),
                "channel_id": ch["id"],
                "video_id": vid["id"],
                "upload_hour_utc": hour,
                "upload_day_of_week": random.randint(0, 6),
                "days_since_last_upload": random.choices([1, 2, 3, 5, 7, 14, 30], weights=[5, 10, 15, 20, 25, 15, 10])[0],
                "hour_drift_from_mean": round(random.gauss(0, 1.5), 2),
                "is_anomalous_time": random.random() < 0.08,
                "created_at": created,
                "updated_at": _updated_at(created),
            })

    def _gen_title_edit_history(self):
        idx = 0
        for vid in random.sample(self._videos, k=min(60, len(self._videos))):
            n_edits = random.choices([0, 1, 2, 3], weights=[40, 35, 20, 5])[0]
            for j in range(n_edits):
                created = _ts(100)
                self.title_edit_history.append({
                    "id": _uid("titleedit", idx),
                    "video_id": vid["id"],
                    "field_name": random.choice(["title", "title", "thumbnail", "description", "tags"]),
                    "old_value": vid.get("title", "Original Title"),
                    "new_value": vid.get("title", "Updated Title") + " [UPDATED]" if random.random() < 0.5 else None,
                    "detected_at": created,
                    "hours_after_upload": round(random.expovariate(0.05), 1),
                    "updated_at": _updated_at(created),
                })
                idx += 1

    def _gen_thumbnail_analyses(self):
        for i, vid in enumerate(self._videos):
            created = vid.get("created_at", _ts())
            colors = [
                {"hex": f"#{random.randint(0,255):02x}{random.randint(0,255):02x}{random.randint(0,255):02x}",
                 "pct": round(random.uniform(0.1, 0.5), 2)}
                for _ in range(random.randint(2, 5))
            ]
            self.thumbnail_analyses.append({
                "id": _uid(f"thumb_{vid['id']}", 0),
                "video_id": vid["id"],
                "channel_id": vid.get("channel_id"),
                "dominant_colors": colors,
                "has_face": random.random() < 0.55,
                "has_text_overlay": random.random() < 0.7,
                "text_content": random.choice(["MUST WATCH", "NEW!", "😱", "EXPLAINED", None, None]),
                "clickbait_score": round(random.betavariate(2, 5), 3),
                "palette_distance_from_channel_mean": round(random.uniform(0, 0.8), 3),
                "brightness": round(random.uniform(0.2, 0.9), 3),
                "saturation": round(random.uniform(0.1, 0.95), 3),
                "contrast": round(random.uniform(0.3, 0.9), 3),
                "created_at": created,
                "updated_at": _updated_at(created),
            })

    def _gen_link_health(self):
        domains = ["github.com", "arxiv.org", "docs.python.org", "stackoverflow.com",
                    "bit.ly", "amzn.to", "patreon.com", "ko-fi.com", "t.me",
                    "discord.gg", "twitter.com", "linkedin.com"]
        idx = 0
        for vid in random.sample(self._videos, k=min(100, len(self._videos))):
            n_links = random.randint(1, 8)
            for j in range(n_links):
                domain = random.choice(domains)
                alive = random.random() < 0.85
                created = vid.get("created_at", _ts())
                self.link_health.append({
                    "id": _uid("linkhealth", idx),
                    "video_id": vid["id"],
                    "url": f"https://{domain}/{hashlib.md5(f'{idx}'.encode()).hexdigest()[:12]}",
                    "domain": domain,
                    "http_status": 200 if alive else random.choice([404, 403, 410, 301, 500]),
                    "is_alive": alive,
                    "is_affiliate": domain in ("amzn.to", "bit.ly") and random.random() < 0.7,
                    "is_self_reference": random.random() < 0.15,
                    "last_checked_at": _ts(10),
                    "first_dead_at": _ts(60) if not alive else None,
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })
                idx += 1

    # ── Comment Archaeology ──────────────────────────────────────────

    def _gen_pinned_comment_history(self):
        idx = 0
        for vid in random.sample(self._videos, k=min(80, len(self._videos))):
            if random.random() < 0.6:  # 60% of videos have a pinned comment
                vid_comments = [c for c in self._comments if c.get("video_id") == vid["id"]]
                created = _ts(300)
                self.pinned_comment_history.append({
                    "id": _uid("pinned", idx),
                    "video_id": vid["id"],
                    "comment_id": vid_comments[0]["id"] if vid_comments else None,
                    "pinned_text": vid_comments[0]["text"][:200] if vid_comments else "Check out the links in the description!",
                    "pinned_by_creator": True,
                    "detected_at": created,
                    "unpinned_at": _ts(60) if random.random() < 0.2 else None,
                    "updated_at": _updated_at(created),
                })
                idx += 1

    def _gen_sentiment_drifts(self):
        windows = ["hour_1", "day_1", "day_7", "day_30", "day_90"]
        idx = 0
        for vid in self._videos[:150]:
            base_sentiment = random.gauss(0.2, 0.2)
            for w in windows:
                drift = random.gauss(0, 0.08) * (windows.index(w) + 1)
                created = vid.get("created_at", _ts())
                avg = round(max(-1, min(1, base_sentiment + drift)), 3)
                pos = round(max(0, 0.5 + avg * 0.4 + random.gauss(0, 0.05)), 3)
                neg = round(max(0, 1 - pos - random.uniform(0, 0.3)), 3)
                self.sentiment_drifts.append({
                    "id": _uid("sentdrift", idx),
                    "video_id": vid["id"],
                    "window_label": w,
                    "avg_sentiment": avg,
                    "comment_count": random.randint(5, 500),
                    "positive_ratio": pos,
                    "negative_ratio": neg,
                    "toxicity_avg": round(random.betavariate(1, 8), 3),
                    "language_distribution": {"en": round(random.uniform(0.5, 0.9), 2), "ar": round(random.uniform(0.05, 0.3), 2)},
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })
                idx += 1

    def _gen_deleted_shadows(self):
        idx = 0
        for vid in random.sample(self._videos, k=min(50, len(self._videos))):
            n = random.randint(0, 5)
            for j in range(n):
                created = _ts(200)
                self.deleted_shadows.append({
                    "id": _uid("shadow", idx),
                    "video_id": vid["id"],
                    "platform_comment_id": f"Ug{hashlib.md5(f'del{idx}'.encode()).hexdigest()[:20]}",
                    "orphaned_reply_count": random.randint(1, 15),
                    "last_known_text": random.choice([
                        "This is wrong because...",
                        "[content captured before deletion]",
                        None, None,
                    ]),
                    "last_known_author": random.choice(self._authors)["display_name"] if random.random() < 0.5 else None,
                    "detected_at": created,
                    "probable_reason": random.choice(["creator_deleted", "spam_filter", "author_deleted", "policy_violation"]),
                    "updated_at": _updated_at(created),
                })
                idx += 1

    # ── Cross-Video Forensics ────────────────────────────────────────

    def _gen_broll_reuses(self):
        if len(self._videos) < 10:
            return
        idx = 0
        for _ in range(min(60, len(self._videos) // 4)):
            v1, v2 = random.sample(self._videos, 2)
            created = _ts(300)
            self.broll_reuses.append({
                "id": _uid("broll", idx),
                "video_id": v1["id"],
                "matched_video_id": v2["id"],
                "timestamp_source": round(random.uniform(5, v1.get("duration_seconds", 300) - 10), 2),
                "timestamp_match": round(random.uniform(5, v2.get("duration_seconds", 300) - 10), 2),
                "duration_seconds": round(random.uniform(2, 15), 2),
                "similarity_score": round(random.uniform(0.75, 0.99), 3),
                "is_stock_footage": random.random() < 0.4,
                "fingerprint_hash": hashlib.md5(f"broll{idx}".encode()).hexdigest(),
                "created_at": created,
                "updated_at": _updated_at(created),
            })
            idx += 1

    def _gen_sponsor_segments(self):
        idx = 0
        types = ["sponsor", "self_promo", "interaction_reminder", "merch"]
        sponsors = ["Brilliant", "NordVPN", "Squarespace", "Skillshare", "ExpressVPN",
                     "Audible", "HelloFresh", "CuriosityStream", "Nebula", "Surfshark"]
        for vid in self._videos:
            if random.random() < 0.35:  # 35% of videos have sponsors
                dur = vid.get("duration_seconds", 300)
                n = random.randint(1, 3)
                for j in range(n):
                    seg_type = random.choice(types)
                    start_ratio = random.choice([0.0, 0.3, 0.5, 0.7, 0.85])
                    start = dur * start_ratio
                    length = random.uniform(15, 90) if seg_type == "sponsor" else random.uniform(5, 30)
                    created = vid.get("created_at", _ts())
                    self.sponsor_segments.append({
                        "id": _uid("sponsor", idx),
                        "video_id": vid["id"],
                        "start_time": round(start, 2),
                        "end_time": round(start + length, 2),
                        "sponsor_name": random.choice(sponsors) if seg_type == "sponsor" else None,
                        "segment_type": seg_type,
                        "confidence": round(random.uniform(0.6, 0.98), 3),
                        "position_ratio": round(start_ratio, 2),
                        "duration_seconds": round(length, 2),
                        "created_at": created,
                        "updated_at": _updated_at(created),
                    })
                    idx += 1

    def _gen_intro_outro_evolutions(self):
        for ch in self._channels:
            ch_vids = sorted(
                [v for v in self._videos if v.get("channel_id") == ch["id"]],
                key=lambda x: x.get("uploaded_at") or x.get("created_at", "")
            )
            has_intro = random.random() < 0.6
            has_outro = random.random() < 0.7
            for j, vid in enumerate(ch_vids):
                # Creators sometimes drop intros mid-career
                if has_intro and j > len(ch_vids) * 0.6 and random.random() < 0.3:
                    has_intro = False
                created = vid.get("created_at", _ts())
                self.intro_outro_evolutions.append({
                    "id": _uid(f"introoutro_{vid['id']}", 0),
                    "channel_id": ch["id"],
                    "video_id": vid["id"],
                    "has_intro": has_intro,
                    "intro_duration_s": round(random.uniform(3, 15), 1) if has_intro else None,
                    "intro_fingerprint": hashlib.md5(f"intro_{ch['id']}".encode()).hexdigest()[:16] if has_intro else None,
                    "has_outro": has_outro,
                    "outro_duration_s": round(random.uniform(8, 30), 1) if has_outro else None,
                    "outro_fingerprint": hashlib.md5(f"outro_{ch['id']}".encode()).hexdigest()[:16] if has_outro else None,
                    "silence_before_ask_ms": random.randint(200, 2500) if random.random() < 0.6 else None,
                    "ask_type": random.choice(["subscribe", "like", "comment", "bell", "none"]),
                    "created_at": created,
                    "updated_at": _updated_at(created),
                })

    # ── Linguistic ───────────────────────────────────────────────────

    def _gen_linguistic_profiles(self):
        for vid in self._videos:
            is_bilingual = vid.get("language") == "ar" or random.random() < 0.15
            created = vid.get("created_at", _ts())
            dur_min = (vid.get("duration_seconds", 300)) / 60

            self.linguistic_profiles.append({
                "id": _uid(f"ling_{vid['id']}", 0),
                "video_id": vid["id"],
                "code_switch_count": random.randint(3, 40) if is_bilingual else 0,
                "code_switch_pairs": [
                    {"from": "en", "to": "ar", "timestamp": round(random.uniform(10, vid.get("duration_seconds", 300) - 10), 1)}
                    for _ in range(random.randint(1, 5))
                ] if is_bilingual else None,
                "primary_language": "ar" if vid.get("language") == "ar" else "en",
                "secondary_language": "en" if vid.get("language") == "ar" else ("ar" if is_bilingual else None),
                "language_ratio": {"en": round(random.uniform(0.5, 0.85), 2), "ar": round(random.uniform(0.1, 0.4), 2)} if is_bilingual else {"en": 0.98},
                "filler_word_count": random.randint(5, int(dur_min * 8)),
                "filler_word_rate": round(random.uniform(1, 12), 2),
                "filler_clusters": [
                    {"timestamp": round(random.uniform(10, vid.get("duration_seconds", 300) - 10), 1),
                     "density": round(random.uniform(0.3, 1.0), 2)}
                    for _ in range(random.randint(1, 5))
                ],
                "hedging_phrases_count": random.randint(2, 30),
                "hedging_rate": round(random.uniform(0.5, 8.0), 2),
                "hedging_by_topic": {
                    random.choice([t["name"] for t in self._topics]): round(random.uniform(0.01, 0.15), 3)
                    for _ in range(random.randint(1, 3))
                },
                "unique_word_count": random.randint(200, 5000),
                "vocabulary_richness": round(random.uniform(0.3, 0.8), 3),
                "zipf_deviation": round(random.gauss(0, 0.15), 4),
                "rare_word_rate": round(random.uniform(0.5, 8.0), 2),
                "avg_sentence_length": round(random.uniform(8, 25), 1),
                "speaking_rate_wpm": round(random.uniform(100, 200), 1),
                "created_at": created,
                "updated_at": _updated_at(created),
            })

    # ── Graph-Relational ─────────────────────────────────────────────

    def _gen_commenter_migrations(self):
        if len(self._authors) < 20 or len(self._channels) < 5:
            return
        idx = 0
        for _ in range(min(100, len(self._authors) // 3)):
            author = random.choice(self._authors)
            ch_from, ch_to = random.sample(self._channels, 2)
            created = _ts(200)
            self.commenter_migrations.append({
                "id": _uid("migration", idx),
                "author_id": author["id"],
                "from_channel_id": ch_from["id"],
                "to_channel_id": ch_to["id"],
                "first_seen_from": _ts(500),
                "last_seen_from": _ts(200),
                "first_seen_to": _ts(180),
                "migration_confidence": round(random.uniform(0.4, 0.95), 3),
                "overlap_period_days": random.randint(0, 60) if random.random() < 0.6 else None,
                "created_at": created,
                "updated_at": _updated_at(created),
            })
            idx += 1

    def _gen_entity_propagations(self):
        idx = 0
        for ent in random.sample(self._entities, k=min(40, len(self._entities))):
            origin_ch = random.choice(self._channels)
            origin_vid = random.choice([v for v in self._videos if v.get("channel_id") == origin_ch["id"]] or self._videos)
            created = _ts(400)
            chain = []
            for step in range(random.randint(1, 8)):
                ch = random.choice(self._channels)
                vid = random.choice([v for v in self._videos if v.get("channel_id") == ch["id"]] or self._videos)
                chain.append({
                    "channel_id": ch["id"],
                    "video_id": vid["id"],
                    "timestamp": _ts(400 - step * 30),
                    "lag_hours": random.randint(12, 720),
                })
            self.entity_propagations.append({
                "id": _uid("entprop", idx),
                "entity_id": ent["id"],
                "origin_channel_id": origin_ch["id"],
                "origin_video_id": origin_vid["id"],
                "origin_timestamp": _ts(500),
                "propagation_chain": chain,
                "total_adopters": len(chain),
                "avg_adoption_lag_hours": round(sum(c["lag_hours"] for c in chain) / len(chain), 1) if chain else 0,
                "created_at": created,
                "updated_at": _updated_at(created),
            })
            idx += 1

    def _gen_parasocial_indices(self):
        idx = 0
        for author in random.sample(self._authors, k=min(120, len(self._authors))):
            ch = random.choice(self._channels)
            n_comments = random.randint(1, 80)
            n_replies = random.choices([0, 0, 0, 1, 2, 3, 5], weights=[40, 20, 10, 10, 10, 5, 5])[0]
            created = _ts(300)
            self.parasocial_indices.append({
                "id": _uid("parasocial", idx),
                "author_id": author["id"],
                "channel_id": ch["id"],
                "author_comments_on_channel": n_comments,
                "creator_replies_to_author": n_replies,
                "reciprocity_ratio": round(n_replies / max(n_comments, 1), 4),
                "first_interaction": _ts(500),
                "last_interaction": _ts(30),
                "engagement_tier": "superfan" if n_comments > 30 else ("regular" if n_comments > 10 else "casual"),
                "created_at": created,
                "updated_at": _updated_at(created),
            })
            idx += 1

    def _gen_topic_gestations(self):
        if not self._entities or not self._channels:
            return
        idx = 0
        for _ in range(min(60, len(self._entities))):
            ent = random.choice(self._entities)
            ch = random.choice(self._channels)
            comment_first = _ts(400)
            has_video = random.random() < 0.6
            created = _ts(200)
            self.topic_gestations.append({
                "id": _uid("gestation", idx),
                "channel_id": ch["id"],
                "entity_id": ent["id"],
                "first_comment_mention_at": comment_first,
                "first_video_mention_at": _ts(200) if has_video else None,
                "gestation_days": random.randint(7, 180) if has_video else None,
                "comment_mention_count_before_video": random.randint(3, 50),
                "audience_requested": random.random() < 0.25,
                "created_at": created,
                "updated_at": _updated_at(created),
            })
            idx += 1
