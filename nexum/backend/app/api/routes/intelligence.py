"""
Nexum API — Intelligence Routes (v5.0).

Exposes all 8 intelligence layers via REST:
  GET  /intelligence/temporal/{video_id}     — Temporal curves + derived metrics
  GET  /intelligence/style/{entity_id}       — Style fingerprint vector
  GET  /intelligence/authenticity/{video_id} — Integrity report
  GET  /intelligence/social/{video_id}       — Social dynamics + community metrics
  GET  /intelligence/intent/{video_id}       — Semantic intent profile
  GET  /intelligence/evolution/{channel_id}  — Cross-video evolution
  GET  /intelligence/graph/{entity_id}       — Graph intelligence
  GET  /intelligence/quality/{video_id}      — Meta-quality report
  GET  /intelligence/full/{video_id}         — All layers combined
  GET  /intelligence/flags                   — Feature flag status
  POST /intelligence/flags                   — Toggle feature flags
"""
from __future__ import annotations

import logging
import random
from typing import Dict, Optional

import traceback
from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/intelligence", tags=["Intelligence"])

# ── Feature Flags ────────────────────────────────────────────────────

_feature_flags: Dict[str, bool] = {
    "temporal_behavior": True,
    "style_fingerprint": True,
    "authenticity": True,
    "social_dynamics": True,
    "semantic_intent": True,
    "cross_video_evolution": True,
    "graph_intelligence": True,
    "meta_quality": True,
}

# ── Lazy Service Initialization ──────────────────────────────────────

_services: Dict = {}


def _get_service(name: str):
    """Lazy-load intelligence services to avoid import cost at startup."""
    if name in _services:
        return _services[name]

    if name == "temporal":
        from app.ml.intelligence.temporal_behavior import TemporalBehaviorService
        _services[name] = TemporalBehaviorService(_feature_flags)
    elif name == "style":
        from app.ml.intelligence.style_fingerprint import StyleFingerprintService
        _services[name] = StyleFingerprintService(_feature_flags)
    elif name == "authenticity":
        from app.ml.intelligence.authenticity import AuthenticityService
        _services[name] = AuthenticityService(_feature_flags)
    elif name == "social":
        from app.ml.intelligence.social_dynamics import SocialDynamicsService
        _services[name] = SocialDynamicsService(_feature_flags)
    elif name == "intent":
        from app.ml.intelligence.semantic_evolution_graph_quality import SemanticIntentService
        _services[name] = SemanticIntentService(_feature_flags)
    elif name == "evolution":
        from app.ml.intelligence.semantic_evolution_graph_quality import CrossVideoEvolutionService
        _services[name] = CrossVideoEvolutionService(_feature_flags)
    elif name == "graph":
        from app.ml.intelligence.semantic_evolution_graph_quality import GraphIntelligenceService
        _services[name] = GraphIntelligenceService(_feature_flags)
    elif name == "quality":
        from app.ml.intelligence.semantic_evolution_graph_quality import MetaQualityService
        _services[name] = MetaQualityService(_feature_flags)

    return _services.get(name)


# ── Demo Data Helper ─────────────────────────────────────────────────

def _get_demo_generator():
    from app.api.routes.demo import _generator
    _generator.generate()
    return _generator


def _find_demo_video(video_id: str):
    gen = _get_demo_generator()
    vid = next((v for v in gen.videos if v["id"] == video_id), None)
    if not vid:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
    return vid, gen


def _find_demo_channel(channel_id: str):
    gen = _get_demo_generator()
    ch = next((c for c in gen.channels if c["id"] == channel_id), None)
    if not ch:
        raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")
    return ch, gen


def _generate_mock_segments(vid: Dict, gen) -> list:
    """Generate mock transcript segments for a demo video."""
    duration = vid.get("duration_seconds") or 300
    segs = []
    t = 0.0
    while t < duration:
        seg_dur = random.uniform(2, 8)
        segs.append({
            "start_time": t, "end_time": min(t + seg_dur, duration),
            "text": f"Mock segment at {t:.0f}s with sample speech content for analysis",
            "confidence": random.uniform(0.7, 0.99),
            "sentiment": random.gauss(0.1, 0.3),
        })
        t += seg_dur
    return segs


def _generate_mock_audio_segments(vid: Dict) -> list:
    """Generate mock audio segments."""
    duration = vid.get("duration_seconds") or 300
    segs = []
    t = 0.0
    while t < duration:
        w = random.uniform(3, 6)
        segs.append({
            "start_time": t, "end_time": min(t + w, duration),
            "loudness_lufs": random.gauss(-18, 4),
            "speech_probability": random.uniform(0.3, 0.95),
            "music_probability": random.uniform(0.0, 0.4),
            "spectral_centroid": random.gauss(2200, 500),
            "dynamic_range_db": random.gauss(12, 4),
            "zero_crossing_rate": random.uniform(0.02, 0.12),
            "harmonic_ratio": random.gauss(0.6, 0.15),
        })
        t += w
    return segs


def _generate_mock_frames(vid: Dict) -> list:
    """Generate mock frame data."""
    duration = vid.get("duration_seconds") or 300
    frames = []
    for i in range(0, int(duration), 5):
        frames.append({
            "timestamp": float(i),
            "visual_tags": random.sample(["person", "text", "screen", "outdoor", "indoor", "chart", "code"], k=random.randint(1, 4)),
            "ocr_text": "sample text" if random.random() < 0.3 else "",
            "ocr_confidence": random.uniform(0.4, 0.95),
            "is_scene_change": random.random() < 0.08,
        })
    return frames


# ── Routes ───────────────────────────────────────────────────────────

@router.get("/flags")
async def get_flags():
    """Get current feature flag status for all intelligence layers."""
    try:
        return {"flags": _feature_flags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.post("/flags")
async def set_flags(flags: Dict[str, bool]):
    """Toggle feature flags. Keys: temporal_behavior, style_fingerprint, authenticity, etc."""
    try:
        for k, v in flags.items():
            if k in _feature_flags:
                _feature_flags[k] = v
        _services.clear()  # force re-init with new flags
        return {"flags": _feature_flags}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/temporal/{video_id}")
async def get_temporal_intelligence(video_id: str):
    """Layer 1: Temporal behavior — curves, silence, derived metrics."""
    try:
        if not _feature_flags.get("temporal_behavior"):
            raise HTTPException(status_code=403, detail="temporal_behavior feature disabled")

        svc = _get_service("temporal")
        vid, gen = _find_demo_video(video_id)
        duration = vid.get("duration_seconds") or 300
        segments = _generate_mock_segments(vid, gen)
        audio_segs = _generate_mock_audio_segments(vid)

        speech_curve = svc.compute_speech_rate_curve(video_id, segments, duration)
        energy_curve = svc.compute_energy_curve(video_id, audio_segs, duration)
        music_curve = svc.compute_music_density_curve(video_id, audio_segs, duration)
        silence = svc.compute_silence_distribution(video_id, audio_segs, duration)
        derived = svc.compute_derived_metrics(video_id, speech_curve, energy_curve, music_curve, duration)

        return {
            "video_id": video_id,
            "layer": "temporal_behavior",
            "speech_rate_curve": {"type": speech_curve.curve_type, "sample_count": len(speech_curve.values),
                                  "mean": speech_curve.mean_val, "max": speech_curve.max_val, "values": speech_curve.values[:100]},
            "energy_curve": {"type": energy_curve.curve_type, "sample_count": len(energy_curve.values),
                             "mean": energy_curve.mean_val, "max": energy_curve.max_val, "values": energy_curve.values[:100]},
            "music_density_curve": {"type": music_curve.curve_type, "mean": music_curve.mean_val, "values": music_curve.values[:100]},
            "silence": {"total_seconds": silence.total_silence_seconds, "ratio": silence.silence_ratio,
                        "gap_count": silence.gap_count, "dramatic_pauses": silence.dramatic_pause_count},
            "derived": {
                "excitement_ramp": derived.excitement_ramp_score,
                "attention_decay": derived.attention_decay_probability,
                "highlights": derived.highlight_timestamps[:10],
                "info_density_per_min": derived.info_density_per_minute,
                "pacing_quality": derived.pacing_quality_score,
                "confidence": derived.confidence,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/style/{entity_id}")
async def get_style_fingerprint(entity_id: str):
    """Layer 2: Style fingerprint — visual, audio, linguistic profiles + 128-dim vector."""
    try:
        if not _feature_flags.get("style_fingerprint"):
            raise HTTPException(status_code=403, detail="style_fingerprint feature disabled")

        svc = _get_service("style")
        gen = _get_demo_generator()

        # Could be video or channel
        vid = next((v for v in gen.videos if v["id"] == entity_id), None)
        ch = next((c for c in gen.channels if c["id"] == entity_id), None)
        if not vid and not ch:
            raise HTTPException(status_code=404, detail="Entity not found")

        target = vid or ch
        frames = _generate_mock_frames(target) if vid else []
        audio_segs = _generate_mock_audio_segments(target) if vid else []
        transcript = " ".join(s["text"] for s in _generate_mock_segments(target, gen)) if vid else "Sample channel description transcript content"

        visual = svc.compute_visual_profile(entity_id, frames, [f for f in frames if f.get("is_scene_change")], target.get("duration_seconds", 300))
        audio = svc.compute_audio_profile(entity_id, audio_segs)
        linguistic = svc.compute_linguistic_profile(entity_id, transcript)
        vector = svc.build_style_vector(entity_id, visual, audio, linguistic)

        return {
            "entity_id": entity_id,
            "layer": "style_fingerprint",
            "visual": {"motion_smoothness": visual.motion_smoothness, "cut_freq": visual.cut_frequency_per_min,
                        "clutter": visual.scene_clutter_index, "text_density": visual.text_density_per_frame},
            "audio": {"loudness_mean": audio.loudness_mean_lufs, "mic_quality": audio.mic_quality_score,
                      "normalization": audio.loudness_normalization_style, "noise_type": audio.noise_type},
            "linguistic": {"vocab_entropy": linguistic.vocabulary_entropy, "formality": linguistic.formality_score,
                           "hedging": linguistic.hedging_frequency, "slang": linguistic.slang_density,
                           "filler_rate": linguistic.filler_word_rate, "ttr": linguistic.type_token_ratio},
            "style_vector": {"dim": len(vector.vector), "confidence": vector.confidence,
                             "vector": vector.vector[:32]},  # truncate for response size
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/authenticity/{video_id}")
async def get_authenticity(video_id: str):
    """Layer 3: Authenticity — manipulation detection with confidence bands."""
    try:
        if not _feature_flags.get("authenticity"):
            raise HTTPException(status_code=403, detail="authenticity feature disabled")

        svc = _get_service("authenticity")
        vid, gen = _find_demo_video(video_id)
        frames = _generate_mock_frames(vid)
        audio_segs = _generate_mock_audio_segments(vid)

        report = svc.analyze_video(video_id, frames, audio_segs)

        def _band(b):
            return {"value": b.value, "lower": b.lower, "upper": b.upper, "method": b.method}

        return {
            "video_id": video_id,
            "layer": "authenticity",
            "overall_integrity": report.overall_integrity_score,
            "risk_level": report.risk_level,
            "cross_modal_consistency": report.cross_modal_consistency,
            "video": {
                "frame_interpolation": _band(report.video_report.frame_interpolation),
                "cgi_probability": _band(report.video_report.cgi_probability),
                "compression_artifacts": _band(report.video_report.compression_artifact_score),
                "lip_sync_mismatch": _band(report.video_report.lip_sync_mismatch),
                "deepfake_probability": _band(report.video_report.deepfake_probability),
                "overall": report.video_report.overall_authenticity,
                "flags": report.video_report.flags,
            },
            "audio": {
                "formant_drift": _band(report.audio_report.formant_drift_anomaly),
                "synthetic_speech": _band(report.audio_report.synthetic_speech_likelihood),
                "over_denoising": _band(report.audio_report.over_denoising_artifact),
                "time_stretch": _band(report.audio_report.time_stretch_distortion),
                "voice_cloning": _band(report.audio_report.voice_cloning_likelihood),
                "overall": report.audio_report.overall_authenticity,
                "flags": report.audio_report.flags,
            },
            "confidence": report.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/social/{video_id}")
async def get_social_dynamics(video_id: str):
    """Layer 4: Social dynamics — debates, community metrics, user profiles."""
    try:
        if not _feature_flags.get("social_dynamics"):
            raise HTTPException(status_code=403, detail="social_dynamics feature disabled")

        svc = _get_service("social")
        vid, gen = _find_demo_video(video_id)
        comments = [c for c in gen.comments if c.get("video_id") == video_id]

        debates = svc.detect_debate_chains(video_id, comments)
        community = svc.compute_community_metrics(video_id, "video", comments)

        return {
            "video_id": video_id,
            "layer": "social_dynamics",
            "debate_chains": [{"root": d.root_comment_id, "participants": d.participant_count,
                               "turns": d.turn_count, "escalation": d.escalation_score,
                               "resolution": d.resolution, "topics": d.key_topics} for d in debates[:10]],
            "community": {
                "polarization": community.polarization_index,
                "meme_velocity": community.meme_velocity,
                "narrative_divergence": community.narrative_divergence_score,
                "toxicity_ratio": community.toxicity_ratio,
                "constructive_ratio": community.constructive_ratio,
                "participants": community.unique_participant_count,
                "avg_thread_depth": community.avg_thread_depth,
                "reciprocity": community.reply_reciprocity,
            },
            "confidence": community.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/intent/{video_id}")
async def get_semantic_intent(video_id: str):
    """Layer 5: Semantic intent — intent, emotion arc, persuasion, bias."""
    try:
        if not _feature_flags.get("semantic_intent"):
            raise HTTPException(status_code=403, detail="semantic_intent feature disabled")

        svc = _get_service("intent")
        vid, gen = _find_demo_video(video_id)
        segments = _generate_mock_segments(vid, gen)
        transcript = " ".join(s["text"] for s in segments)

        profile = svc.analyze(video_id, transcript, segments, vid.get("duration_seconds", 300))

        return {
            "video_id": video_id,
            "layer": "semantic_intent",
            "intent_scores": profile.intent_scores,
            "primary_intent": profile.primary_intent,
            "emotional_arc": {"type": profile.emotional_arc_type,
                              "points": profile.emotional_arc_points[:20]},
            "persuasion": {"style": profile.persuasion_style, "intensity": profile.persuasion_intensity},
            "certainty_ratio": profile.certainty_ratio,
            "bias_framing": profile.bias_framing_polarity,
            "rhetorical_question_density": profile.rhetorical_question_density,
            "cta_count": profile.call_to_action_count,
            "confidence": profile.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/evolution/{channel_id}")
async def get_evolution(channel_id: str):
    """Layer 6: Cross-video evolution — longitudinal creator tracking."""
    try:
        if not _feature_flags.get("cross_video_evolution"):
            raise HTTPException(status_code=403, detail="cross_video_evolution feature disabled")

        svc = _get_service("evolution")
        ch, gen = _find_demo_channel(channel_id)
        videos = sorted([v for v in gen.videos if v.get("channel_id") == channel_id],
                        key=lambda v: v.get("created_at", ""))

        profile = svc.compute_evolution(channel_id, videos)

        return {
            "channel_id": channel_id,
            "layer": "cross_video_evolution",
            "video_count": profile.video_count,
            "time_range": profile.time_range,
            "drifts": {
                "intro_length": profile.intro_length_drift,
                "speech_speed": profile.speech_speed_drift,
                "vocabulary_growth": profile.vocabulary_growth_rate,
                "sponsor_density": profile.sponsor_density_trend,
                "editing_complexity": profile.editing_complexity_trend,
            },
            "production_value_curve": profile.production_value_curve,
            "phases": profile.phases,
            "confidence": profile.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/graph/{entity_id}")
async def get_graph_intelligence(entity_id: str):
    """Layer 7: Graph-derived intelligence — centrality, bridges, propagation."""
    try:
        if not _feature_flags.get("graph_intelligence"):
            raise HTTPException(status_code=403, detail="graph_intelligence feature disabled")

        svc = _get_service("graph")
        gen = _get_demo_generator()
        snapshot = gen.generate()

        results = svc.compute_centralities(snapshot["nodes"], snapshot["edges"])
        intel = results.get(entity_id)
        if not intel:
            raise HTTPException(status_code=404, detail="Entity not found in graph")

        return {
            "entity_id": entity_id,
            "layer": "graph_intelligence",
            "entity_type": intel.entity_type,
            "influence_centrality": intel.influence_centrality,
            "betweenness_centrality": intel.betweenness_centrality,
            "is_bridge_node": intel.is_bridge_node,
            "reciprocity_imbalance": intel.reciprocity_imbalance,
            "confidence": intel.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/quality/{video_id}")
async def get_meta_quality(video_id: str):
    """Layer 8: Meta-quality — system-level reliability assessment."""
    try:
        if not _feature_flags.get("meta_quality"):
            raise HTTPException(status_code=403, detail="meta_quality feature disabled")

        svc = _get_service("quality")
        vid, gen = _find_demo_video(video_id)
        segments = _generate_mock_segments(vid, gen)
        frames = _generate_mock_frames(vid)
        audio = _generate_mock_audio_segments(vid)
        comments = [c for c in gen.comments if c.get("video_id") == video_id]

        report = svc.evaluate(video_id, segments, frames, audio, comments, vid.get("duration_seconds", 300))

        return {
            "video_id": video_id,
            "layer": "meta_quality",
            "transcript_reliability": report.transcript_reliability_index,
            "visual_clarity": report.visual_clarity_index,
            "audio_cleanliness": report.audio_cleanliness_score,
            "editing_professionalism": report.editing_professionalism_score,
            "redundancy": report.redundancy_index,
            "information_density": report.information_density_score,
            "overall_quality": report.overall_quality,
            "data_completeness": report.data_completeness,
            "confidence": report.confidence,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")


@router.get("/full/{video_id}")
async def get_full_intelligence(video_id: str):
    """All 8 layers combined for a single video. Lazy-computes only enabled layers."""
    try:
        results = {"video_id": video_id, "layers": {}}

        for layer_name, route_fn in [
            ("temporal_behavior", get_temporal_intelligence),
            ("authenticity", get_authenticity),
            ("semantic_intent", get_semantic_intent),
            ("social_dynamics", get_social_dynamics),
            ("meta_quality", get_meta_quality),
        ]:
            if _feature_flags.get(layer_name, False):
                try:
                    results["layers"][layer_name] = await route_fn(video_id)
                except HTTPException:
                    results["layers"][layer_name] = {"error": "unavailable"}

        # Style and evolution need entity_id logic
        if _feature_flags.get("style_fingerprint"):
            try:
                results["layers"]["style_fingerprint"] = await get_style_fingerprint(video_id)
            except HTTPException:
                results["layers"]["style_fingerprint"] = {"error": "unavailable"}

        results["enabled_layers"] = [k for k, v in _feature_flags.items() if v]
        results["disabled_layers"] = [k for k, v in _feature_flags.items() if not v]
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Intelligence route error: {e}")
        raise HTTPException(status_code=500, detail=f"Intelligence computation failed: {type(e).__name__}")
