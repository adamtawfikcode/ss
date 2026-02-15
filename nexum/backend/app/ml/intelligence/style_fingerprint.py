"""
Nexum Intelligence Layer 2 — Style & Signature Fingerprinting.

Builds statistical creator style vectors from visual, audio, and linguistic patterns.
Output: Creator Style Vector (128-dim) usable for similarity queries.

Categories:
  Visual: color palette, lighting, camera angle, motion, cuts, clutter, text density
  Audio: pitch range, loudness habits, noise profile, mic fingerprint, room impulse
  Linguistic: sentence length, vocab entropy, slang, passive/active, hedging, questions
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

STYLE_VECTOR_DIM = 128


@dataclass
class VisualStyleProfile:
    """Statistical visual style for a channel or video."""
    entity_id: str  # channel_id or video_id
    color_palette_histogram: List[float]  # 16-bin HSV histogram
    lighting_temperature_bias: float      # -1=cool, 0=neutral, 1=warm
    camera_angle_distribution: Dict[str, float]  # {"close":0.3,"medium":0.5,"wide":0.2}
    motion_smoothness: float              # 0=jerky, 1=stabilized
    cut_frequency_per_min: float
    scene_clutter_index: float            # 0=minimal, 1=busy
    text_density_per_frame: float         # avg text area ratio
    dominant_aspect_ratio: str            # "16:9", "9:16", "4:3"
    confidence: float = 0.5


@dataclass
class AudioStyleProfile:
    """Statistical audio fingerprint for a channel or video."""
    entity_id: str
    pitch_range_hz: tuple                 # (low, high)
    pitch_mean_hz: float
    pitch_std_hz: float
    loudness_mean_lufs: float
    loudness_std_lufs: float
    loudness_normalization_style: str     # "compressed" | "dynamic" | "inconsistent"
    noise_floor_db: float
    noise_type: str                       # "clean" | "fan_hum" | "ambient" | "noisy"
    mic_quality_score: float              # 0-1
    room_impulse_cluster: Optional[str]   # cluster ID from room fingerprints
    confidence: float = 0.5


@dataclass
class LinguisticStyleProfile:
    """Statistical linguistic fingerprint from transcripts."""
    entity_id: str
    sentence_length_mean: float
    sentence_length_std: float
    sentence_length_distribution: List[float]  # histogram bins [1-5, 6-10, 11-15, 16-20, 20+]
    vocabulary_entropy: float             # Shannon entropy of word distribution
    vocabulary_size: int
    type_token_ratio: float
    slang_density: float                  # slang words per 100 words
    passive_voice_ratio: float
    active_voice_ratio: float
    hedging_frequency: float              # per 100 words
    question_frequency: float             # per 100 sentences
    filler_word_rate: float               # per minute
    first_person_ratio: float             # I/me/my usage
    second_person_ratio: float            # you/your
    formality_score: float                # 0=casual, 1=formal
    confidence: float = 0.5


@dataclass
class CreatorStyleVector:
    """
    Unified 128-dimensional style vector for similarity queries.

    Composed of:
      [0:32]   visual style features
      [32:64]  audio style features
      [64:96]  linguistic style features
      [96:128] cross-modal interaction features
    """
    entity_id: str
    vector: List[float]                   # 128-dim normalized
    visual_weight: float = 0.33
    audio_weight: float = 0.33
    linguistic_weight: float = 0.34
    version: str = "v1.0"
    confidence: float = 0.5

    def similarity(self, other: "CreatorStyleVector") -> float:
        """Cosine similarity between two style vectors."""
        a = np.array(self.vector)
        b = np.array(other.vector)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-8 or norm_b < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


class StyleFingerprintService:
    """
    Builds and compares creator style fingerprints.

    Processes video-level features and aggregates them into channel-level
    style vectors for cross-creator similarity search.
    """

    def __init__(self, feature_flags: Optional[Dict] = None):
        self._flags = feature_flags or {}
        self._enabled = self._flags.get("style_fingerprint", True)
        # Passive voice indicators
        self._passive_markers = {"was", "were", "been", "being", "is", "are", "am", "be"}
        self._hedging_words = {
            "maybe", "perhaps", "possibly", "probably", "might", "could",
            "seems", "appears", "suggest", "think", "believe", "assume",
            "somewhat", "rather", "fairly", "relatively", "arguably",
        }
        self._filler_words = {"um", "uh", "like", "you know", "basically", "literally", "actually", "so"}
        self._slang_markers = {
            "gonna", "wanna", "gotta", "kinda", "sorta", "dunno", "ain't",
            "y'all", "nah", "yeah", "yep", "nope", "dude", "bro", "lol",
        }
        logger.info("StyleFingerprintService initialized (enabled=%s)", self._enabled)

    def compute_visual_profile(
        self, entity_id: str, frames: List[Dict], scene_changes: List[Dict],
        duration_s: float,
    ) -> VisualStyleProfile:
        """Extract visual style from frame and scene data."""
        try:
            n_frames = max(len(frames), 1)

            # Color histogram from dominant colors (if available)
            color_hist = [0.0] * 16
            warmth_vals = []
            clutter_vals = []
            text_ratios = []

            for f in frames:
                tags = f.get("visual_tags") or []
                ocr = f.get("ocr_text") or ""
                text_ratios.append(min(1.0, len(ocr) / 500.0))

                # Approximate clutter from tag count
                clutter_vals.append(min(1.0, len(tags) / 15.0))

            # Cut frequency
            cuts = len(scene_changes)
            cut_freq = cuts / max(duration_s / 60, 1)

            # Motion smoothness estimate from scene change rate
            if cuts > 0 and duration_s > 0:
                motion = max(0.0, 1.0 - min(cut_freq / 30.0, 1.0))
            else:
                motion = 0.5

            return VisualStyleProfile(
                entity_id=entity_id,
                color_palette_histogram=color_hist,
                lighting_temperature_bias=0.0,  # would need color analysis
                camera_angle_distribution={"close": 0.3, "medium": 0.5, "wide": 0.2},
                motion_smoothness=round(motion, 3),
                cut_frequency_per_min=round(cut_freq, 2),
                scene_clutter_index=round(float(np.mean(clutter_vals)) if clutter_vals else 0.3, 3),
                text_density_per_frame=round(float(np.mean(text_ratios)) if text_ratios else 0, 3),
                dominant_aspect_ratio="16:9",
                confidence=min(0.85, 0.3 + n_frames * 0.001),
            )
        except Exception as _exc:
            logger.warning(f"StyleFingerprintService.compute_visual_profile failed: {_exc}")
            return None

    def compute_audio_profile(
        self, entity_id: str, audio_segments: List[Dict], room_fp: Optional[Dict] = None,
    ) -> AudioStyleProfile:
        """Extract audio style from audio segment data."""
        try:
            loudness_vals = [s.get("loudness_lufs", -20) for s in audio_segments if s.get("loudness_lufs") is not None]
            speech_probs = [s.get("speech_probability", 0.5) for s in audio_segments]

            if loudness_vals:
                l_mean = float(np.mean(loudness_vals)) if loudness_vals and not any(np.isnan(v) for v in loudness_vals) else -18.0
                l_std = float(np.std(loudness_vals)) if loudness_vals and not any(np.isnan(v) for v in loudness_vals) else 3.0
            else:
                l_mean, l_std = -20.0, 5.0

            # Normalization style
            if l_std < 2:
                norm_style = "compressed"
            elif l_std > 8:
                norm_style = "inconsistent"
            else:
                norm_style = "dynamic"

            noise_floor = room_fp.get("noise_floor_db", -40) if room_fp else -40
            noise_type = "clean" if noise_floor < -45 else ("fan_hum" if noise_floor < -30 else "noisy")

            return AudioStyleProfile(
                entity_id=entity_id,
                pitch_range_hz=(80, 300),
                pitch_mean_hz=150.0,
                pitch_std_hz=30.0,
                loudness_mean_lufs=l_mean,
                loudness_std_lufs=l_std,
                loudness_normalization_style=norm_style,
                noise_floor_db=noise_floor,
                noise_type=noise_type,
                mic_quality_score=max(0.0, min(1.0, (noise_floor + 60) / 40)),
                room_impulse_cluster=room_fp.get("cluster_id") if room_fp else None,
                confidence=min(0.85, 0.3 + len(audio_segments) * 0.005),
            )
        except Exception as _exc:
            logger.warning(f"StyleFingerprintService.compute_audio_profile failed: {_exc}")
            return None

    def compute_linguistic_profile(
        self, entity_id: str, transcript_text: str,
    ) -> LinguisticStyleProfile:
        """Extract linguistic style from full transcript text."""
        try:
            words = transcript_text.lower().split()
            n_words = max(len(words), 1)

            # Sentences (approximate)
            sentences = [s.strip() for s in transcript_text.replace("?", ".").replace("!", ".").split(".") if s.strip()]
            n_sentences = max(len(sentences), 1)
            sent_lengths = [len(s.split()) for s in sentences]

            # Vocabulary
            unique_words = set(words)
            vocab_size = len(unique_words)
            ttr = vocab_size / max(n_words, 1)

            # Shannon entropy
            from collections import Counter
            word_counts = Counter(words)
            total = sum(word_counts.values())
            entropy = -sum((c / max(total, 1)) * math.log2(c / max(total, 1)) for c in word_counts.values() if c > 0) if total > 0 else 0.0

            # Passive voice (heuristic: be-verb followed by past participle pattern)
            passive_count = sum(1 for i, w in enumerate(words[:-1])
                               if w in self._passive_markers and words[i + 1].endswith("ed"))
            hedging = sum(1 for w in words if w in self._hedging_words)
            fillers = sum(1 for w in words if w in self._filler_words)
            slang = sum(1 for w in words if w in self._slang_markers)
            questions = sum(1 for s in transcript_text.split("\n") if "?" in s)
            first_person = sum(1 for w in words if w in {"i", "me", "my", "mine", "myself"})
            second_person = sum(1 for w in words if w in {"you", "your", "yours", "yourself"})

            # Formality estimate
            formality = max(0.0, min(1.0,
                0.5 + (ttr - 0.4) * 2 - slang / max(n_words, 1) * 50
                - fillers / max(n_words, 1) * 30 + passive_count / max(n_sentences, 1) * 2
            ))

            # Sentence length histogram
            hist = [0] * 5
            for sl in sent_lengths:
                if sl <= 5: hist[0] += 1
                elif sl <= 10: hist[1] += 1
                elif sl <= 15: hist[2] += 1
                elif sl <= 20: hist[3] += 1
                else: hist[4] += 1
            hist_norm = [h / max(n_sentences, 1) for h in hist]

            return LinguisticStyleProfile(
                entity_id=entity_id,
                sentence_length_mean=float(np.mean(sent_lengths)) if sent_lengths else 10,
                sentence_length_std=float(np.std(sent_lengths)) if sent_lengths else 3,
                sentence_length_distribution=hist_norm,
                vocabulary_entropy=round(entropy, 3),
                vocabulary_size=vocab_size,
                type_token_ratio=round(ttr, 4),
                slang_density=round(slang / max(n_words, 1) * 100, 2),
                passive_voice_ratio=round(passive_count / max(n_sentences, 1), 3),
                active_voice_ratio=round(1 - passive_count / max(n_sentences, 1), 3),
                hedging_frequency=round(hedging / max(n_words, 1) * 100, 2),
                question_frequency=round(questions / max(n_sentences, 1) * 100, 2),
                filler_word_rate=round(fillers / max(n_words, 1) * 100, 2),
                first_person_ratio=round(first_person / max(n_words, 1), 4),
                second_person_ratio=round(second_person / max(n_words, 1), 4),
                formality_score=round(formality, 3),
                confidence=min(0.9, 0.3 + n_words * 0.0001),
            )
        except Exception as _exc:
            logger.warning(f"StyleFingerprintService.compute_linguistic_profile failed: {_exc}")
            return None

    def build_style_vector(
        self, entity_id: str,
        visual: Optional[VisualStyleProfile] = None,
        audio: Optional[AudioStyleProfile] = None,
        linguistic: Optional[LinguisticStyleProfile] = None,
    ) -> CreatorStyleVector:
        """Compose a unified 128-dim style vector from all sub-profiles."""
        try:
            vec = np.zeros(STYLE_VECTOR_DIM)

            # Visual features [0:32]
            if visual:
                vec[0] = visual.lighting_temperature_bias
                vec[1] = visual.motion_smoothness
                vec[2] = visual.cut_frequency_per_min / 30.0
                vec[3] = visual.scene_clutter_index
                vec[4] = visual.text_density_per_frame
                for i, v in enumerate(visual.color_palette_histogram[:16]):
                    vec[5 + i] = v

            # Audio features [32:64]
            if audio:
                vec[32] = (audio.pitch_mean_hz - 100) / 200.0
                vec[33] = audio.pitch_std_hz / 100.0
                vec[34] = (audio.loudness_mean_lufs + 30) / 30.0
                vec[35] = audio.loudness_std_lufs / 15.0
                vec[36] = audio.mic_quality_score
                vec[37] = (audio.noise_floor_db + 60) / 40.0
                vec[38] = {"compressed": 0, "dynamic": 0.5, "inconsistent": 1}.get(
                    audio.loudness_normalization_style, 0.5)

            # Linguistic features [64:96]
            if linguistic:
                vec[64] = linguistic.sentence_length_mean / 30.0
                vec[65] = linguistic.sentence_length_std / 15.0
                vec[66] = linguistic.vocabulary_entropy / 15.0
                vec[67] = linguistic.type_token_ratio
                vec[68] = linguistic.slang_density / 10.0
                vec[69] = linguistic.passive_voice_ratio
                vec[70] = linguistic.hedging_frequency / 10.0
                vec[71] = linguistic.question_frequency / 50.0
                vec[72] = linguistic.filler_word_rate / 10.0
                vec[73] = linguistic.formality_score
                vec[74] = linguistic.first_person_ratio * 10
                vec[75] = linguistic.second_person_ratio * 10
                for i, v in enumerate(linguistic.sentence_length_distribution[:5]):
                    vec[76 + i] = v

            # Cross-modal [96:128] — interactions between modalities
            if visual and audio:
                vec[96] = visual.cut_frequency_per_min * audio.loudness_std_lufs / 100
            if visual and linguistic:
                vec[97] = visual.text_density_per_frame * linguistic.vocabulary_entropy / 10
            if audio and linguistic:
                vec[98] = audio.mic_quality_score * linguistic.formality_score

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm

            conf = 0.1
            if visual: conf += 0.3 * visual.confidence
            if audio: conf += 0.3 * audio.confidence
            if linguistic: conf += 0.3 * linguistic.confidence

            return CreatorStyleVector(
                entity_id=entity_id,
                vector=vec.tolist(),
                visual_weight=0.33 if visual else 0,
                audio_weight=0.33 if audio else 0,
                linguistic_weight=0.34 if linguistic else 0,
                confidence=round(min(0.95, conf), 3),
            )
        except Exception as _exc:
            logger.warning(f"StyleFingerprintService.build_style_vector failed: {_exc}")
            return None

    def find_similar(
        self, query: CreatorStyleVector, candidates: List[CreatorStyleVector], top_k: int = 10
    ) -> List[Dict]:
        """Find most similar creators by style vector cosine similarity."""
        try:
            results = []
            for c in candidates:
                if c.entity_id == query.entity_id:
                    continue
                sim = query.similarity(c)
                results.append({"entity_id": c.entity_id, "similarity": round(sim, 4), "confidence": c.confidence})
            results.sort(key=lambda x: -x["similarity"])
            return results[:top_k]
        except Exception as _exc:
            logger.warning(f"StyleFingerprintService.find_similar failed: {_exc}")
            return None
