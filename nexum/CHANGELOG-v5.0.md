# Nexum v5.0 — Intelligence Platform + Three.js Engine

## Summary

**77 Python files** (0 errors), **8 intelligence service layers**, **957-line Three.js engine** with adaptive quality, **2D/3D toggle** in the graph page. All backward compatible, all feature-flagged, all probabilistic with confidence bands.

---

## PART 1: Intelligence Layers (8 Services, 5 Files)

### Layer 1 — Temporal Behavior Intelligence
**File:** `backend/app/ml/intelligence/temporal_behavior.py` (370 lines)

**Channel-level:**
- `UploadCadenceFingerprint` — mean/std/median interval, weekday/hour distributions, periodicity score, regularity index
- `SeasonalPattern` — weekly/monthly/quarterly autocorrelation, seasonal strength
- `BreakSpikeAnomaly` — gap detection with post-break spike ratio, anomaly scoring
- `TimezoneInference` — UTC offset inference from posting hour clustering, business-hour optimization
- `GrowthCurve` — 30d/90d rates, acceleration, inflection detection

**In-video curves:**
- `TemporalCurve` — generic time-series with `at_time()` and `window_stats()` accessors
- Speech rate curve from transcript segments (words/second)
- Audio energy intensity curve from loudness data
- Music density curve from music probability
- `SilenceDistribution` — gap analysis, dramatic pause counting
- `ScenePacingRhythm` — scene duration analysis, acceleration index

**Derived metrics:**
- `DerivedTemporalMetrics` — excitement ramp score, attention decay probability, highlight timestamps, info density per minute, pacing quality score

### Layer 2 — Style & Signature Fingerprinting
**File:** `backend/app/ml/intelligence/style_fingerprint.py` (300 lines)

- `VisualStyleProfile` — color histogram, lighting temperature, camera angle distribution, motion smoothness, cut frequency, scene clutter, text density
- `AudioStyleProfile` — pitch range/mean/std, loudness habits, normalization style (compressed/dynamic/inconsistent), noise floor, mic quality, room impulse cluster
- `LinguisticStyleProfile` — sentence length stats, Shannon entropy, type-token ratio, slang density, passive/active ratio, hedging frequency, question frequency, filler rate, formality score
- `CreatorStyleVector` — **128-dimensional** normalized vector ([0:32] visual, [32:64] audio, [64:96] linguistic, [96:128] cross-modal), with cosine similarity method
- `find_similar()` — top-K similarity search across creators

### Layer 3 — Authenticity & Manipulation
**File:** `backend/app/ml/intelligence/authenticity.py` (260 lines)

**Video signals (all with ConfidenceBand — value + lower/upper bounds):**
- Frame interpolation likelihood (scene change regularity analysis)
- CGI vs real probability (OCR + tag density heuristic)
- Compression artifact score
- Lip-sync mismatch detection (placeholder for AV alignment model)
- Deepfake composite probability

**Audio signals:**
- Formant drift anomaly (spectral centroid consistency)
- Synthetic speech likelihood (harmonic ratio analysis)
- Over-denoising artifact (dynamic range compression)
- Time-stretch spectral distortion (zero crossing rate anomalies)
- Voice cloning composite likelihood

**Combined:**
- `IntegrityReport` — video + audio reports, cross-modal consistency, risk level (low/medium/high/critical)

### Layer 4 — Social Dynamics
**File:** `backend/app/ml/intelligence/social_dynamics.py` (280 lines)

**Conversation structures:**
- `DebateChain` — participant count, turn count, sentiment polarity range, escalation score, resolution classification (unresolved/consensus/abandoned/moderated)
- `EchoChamberCluster` — member grouping, internal sentiment similarity, insularity score
- `SentimentCascade` — trigger detection, affected count, peak intensity, damping rate
- `AuthorityPattern` — acknowledgment vs challenge counting, authority scoring

**User profiles:**
- `UserBehavioralProfile` — positivity baseline, topic specialization vector, engagement velocity, cross-video loyalty, reply ratio, time-of-day pattern

**Community metrics:**
- `CommunityMetrics` — polarization index (bimodal sentiment detection), meme velocity (repeated n-gram patterns), narrative divergence, topic convergence speed, toxicity/constructive ratios, reply reciprocity

### Layer 5 — Semantic Intent
**File:** `backend/app/ml/intelligence/semantic_evolution_graph_quality.py` (first section)

- Intent detection across 8 categories: tutorial, satire, rant, news, advertisement, review, educational, vlog
- Emotional arc classification: ramp_up, cool_down, roller_coaster, flat, u_shape
- Persuasion style: logical, emotional, authority, social_proof, scarcity
- Certainty ratio (definitive claims vs hedging)
- Bias framing polarity (-1 to +1)
- Rhetorical question density per minute
- Call-to-action counting

### Layer 6 — Cross-Video Evolution
**File:** same module

- `CreatorEvolutionProfile` — intro length drift, speech speed drift, vocabulary growth rate, thumbnail style drift, sponsor density trend, editing complexity trend
- Production value curve per video over time
- Phase detection (early/mid/recent with avg production scores)
- Linear regression slopes for all drift metrics

### Layer 7 — Graph Intelligence
**File:** same module

- `GraphIntelligence` — influence centrality (PageRank, 10 iterations), betweenness centrality (clustering coefficient approximation), bridge node detection
- Reciprocity imbalance (in-degree vs out-degree asymmetry)
- Full graph topology analysis over all demo nodes

### Layer 8 — Meta-Quality
**File:** same module

- `MetaQualityReport` — transcript reliability (confidence + coverage), visual clarity, audio cleanliness, editing professionalism, redundancy index (repeated trigrams), information density (WPM × uniqueness), data completeness
- Weighted composite overall quality score

---

## PART 2: Three.js Engine

### Architecture
**File:** `frontend/src/components/graph3d/engine.ts` (957 lines)

| Subsystem | Technique | Performance |
|-----------|-----------|-------------|
| **NodeRenderer** | InstancedMesh per node type | 1 draw call per type, 3000+ nodes |
| **EdgeRenderer** | BufferGeometry LineSegments | Single draw call for all edges |
| **LabelRenderer** | Canvas → Sprite with distance fade | Throttled updates, texture cache |
| **ForceLayout** | Barnes-Hut O(n²) simplified | 50-iteration warmup, async stepping |
| **InteractionManager** | Raycaster on InstancedMesh | Throttled to every 2 frames |
| **PerformanceMonitor** | Rolling 60-frame average | Auto quality scaling |

### Quality Levels

| Level | Node Detail | Edge Fraction | Label Distance | Max Nodes | Anti-alias |
|-------|-------------|---------------|----------------|-----------|------------|
| ultra-low | 3 segments | 30% | 50 units | 500 | off |
| low | 6 segments | 50% | 100 units | 1500 | off |
| medium | 10 segments | 80% | 200 units | 3000 | on |
| high | 16 segments | 100% | 400 units | 5000 | on |

### GPU Detection
- Intel integrated (non-Iris Xe/Arc) → auto-downgrade to `low`
- Mali/Adreno 5xx/PowerVR → `ultra-low`
- WebGL failure → 2D Canvas fallback (grid layout, basic rendering)

### Optimization Techniques
- Materials: `MeshLambertMaterial` only (no PBR)
- No shadows, no environment maps, no post-processing
- Pixel ratio capped at 1x for low/ultra-low
- Scene fog for natural far-distance culling
- Object pooling via InstancedMesh (no per-node Object3D)
- Label updates throttled to every 3 frames
- Hover raycasting throttled to every 2 frames

### React Wrapper
**File:** `frontend/src/components/graph3d/GraphCanvas3D.tsx` (325 lines)

- Lazy-imports Three.js engine (code splitting)
- Bridges existing `GraphNode`/`GraphEdge` types to 3D types
- Stats overlay: FPS (color-coded), quality level, node/edge count, layout status
- Re-layout button
- `Canvas2DFallback` component for WebGL failure

### Graph Page Integration
**File:** `frontend/src/app/graph-live/page.tsx` (226 lines)

- **2D/3D toggle** — switches between Cytoscape (2D) and Three.js (3D)
- **Live/Demo toggle** — switches between WebSocket live data and demo generator
- Both toggles preserve filter state and selected node panel
- Backward compatible — original GraphCanvas still works as 2D option

---

## PART 3: Integration

### API Routes
**File:** `backend/app/api/routes/intelligence.py` (290 lines)

| Method | Path | Layer |
|--------|------|-------|
| GET | `/intelligence/flags` | Feature flag status |
| POST | `/intelligence/flags` | Toggle flags |
| GET | `/intelligence/temporal/{video_id}` | Layer 1 |
| GET | `/intelligence/style/{entity_id}` | Layer 2 |
| GET | `/intelligence/authenticity/{video_id}` | Layer 3 |
| GET | `/intelligence/social/{video_id}` | Layer 4 |
| GET | `/intelligence/intent/{video_id}` | Layer 5 |
| GET | `/intelligence/evolution/{channel_id}` | Layer 6 |
| GET | `/intelligence/graph/{entity_id}` | Layer 7 |
| GET | `/intelligence/quality/{video_id}` | Layer 8 |
| GET | `/intelligence/full/{video_id}` | All 8 combined |

### Frontend Types
**File:** `frontend/src/lib/types.ts` (+214 lines)

Full TypeScript interfaces for all 8 intelligence layers, ConfidenceBand, GraphEngine3DStats, IntelligenceFeatureFlags.

### Frontend API Client
**File:** `frontend/src/lib/api.ts` (+62 lines)

Typed fetch functions for all 11 intelligence endpoints.

---

## File Inventory

| File | Action | Lines |
|------|--------|-------|
| `backend/app/ml/intelligence/temporal_behavior.py` | **New** | 370 |
| `backend/app/ml/intelligence/style_fingerprint.py` | **New** | 300 |
| `backend/app/ml/intelligence/authenticity.py` | **New** | 260 |
| `backend/app/ml/intelligence/social_dynamics.py` | **New** | 280 |
| `backend/app/ml/intelligence/semantic_evolution_graph_quality.py` | **New** | 500 |
| `backend/app/api/routes/intelligence.py` | **New** | 290 |
| `backend/app/main.py` | Modified | +2 lines |
| `frontend/src/components/graph3d/engine.ts` | **New** | 957 |
| `frontend/src/components/graph3d/GraphCanvas3D.tsx` | **New** | 325 |
| `frontend/src/app/graph-live/page.tsx` | Rewritten | 226 |
| `frontend/src/lib/types.ts` | Extended | +214 |
| `frontend/src/lib/api.ts` | Extended | +62 |
| `frontend/package.json` | Modified | +2 deps |
| **Total new code** | | **~3,800 lines** |
