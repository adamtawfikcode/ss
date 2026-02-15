# Nexum v4.2 — Demo Mode, Priority Queue & Input Bar

## New Features

### 1. Demo Mode — Massive Realistic Graph Network
Generates a fully interconnected demo dataset on-demand with **3,069+ nodes** and **10,992+ edges** across all 8 node types:

| Node Type | Count | Examples |
|-----------|-------|---------|
| Channel | 32 | Bayyinah Institute, Fireship, Kurzgesagt, Al Jazeera, 3Blue1Brown |
| Video | 240 | "Linguistic Miracles of Surah Al-Kahf", "Rust in 100 Seconds", "The Math Behind Gradient Descent" |
| Comment | 1,800 | Realistic YouTube comments with reply threads, sentiment, entities |
| CommentAuthor | 350 | Diverse names (Arabic, English, Asian, European) |
| Entity | 85 | People (Linus Torvalds, Ibn Kathir), Technologies (PyTorch, Neo4j), Organizations, Concepts |
| Topic | 42 | "Quranic Linguistics", "Machine Learning", "Linux Administration", "Arabic Calligraphy" |
| Playlist | 20 | "Quran Tafsir Series", "Machine Learning Bootcamp", "Rust Programming" |
| Segment | 500+ | Transcript segments per video |

**12 edge types**: UPLOADED, WROTE, ON, REPLIES_TO, MENTIONS, DISCUSSES, APPEARS_IN, CONTAINS, HAS_PLAYLIST, CO_OCCURS_WITH, SHARES_AUDIENCE_WITH, COMMENTED_ON

**API Endpoints:**
- `GET /api/v1/demo/snapshot` — Full graph snapshot (filterable by node type)
- `GET /api/v1/demo/stats` — Node counts + edge distribution
- `POST /api/v1/demo/nodes` — Create new demo node (any type, auto-connected)
- `GET /api/v1/demo/nodes/{type}` — List nodes by type with pagination
- `POST /api/v1/demo/reset` — Regenerate with new seed

### 2. Priority Queue System — Interruptible Task Pipeline
Two-tier queue with **pause/resume/stack** semantics:

```
┌─────────────────────────────────────────┐
│  Priority Stack (LIFO)                  │  ← drains first
│  [job_C] [job_B]                        │
├─────────────────────────────────────────┤
│  Normal Queue (FIFO)                    │  ← when stack empty
│  [job_1] → [job_2] → [job_3]           │
└─────────────────────────────────────────┘
│  Currently Running: job_A (may be paused) │
└───────────────────────────────────────────┘
```

**Behavior:**
- Normal mode: Jobs process in FIFO order
- Immediate/Priority mode: Current job gets **paused at the next stage boundary**, priority job runs, then paused job **resumes exactly where it left off**
- Stackable: Multiple priority jobs form a LIFO stack — each interrupts the previous
- 9-stage video processing pipeline: metadata → download → transcribe → frames → OCR → embeddings → audio → graph → index

**API Endpoints:**
- `POST /api/v1/queue/enqueue` — Add URL with `priority: "normal" | "immediate"`
- `GET /api/v1/queue/status` — Full queue state (current, stack, queue, paused, completed)
- `GET /api/v1/queue/job/{id}` — Single job status
- `POST /api/v1/queue/cancel/{id}` — Cancel job
- `WS /api/v1/queue/ws/queue` — Real-time queue events

### 3. URL Input Bar (Frontend)
- Auto-detects URL type (video/channel/playlist) from URL patterns
- Shows detected type badge
- **Priority toggle**: Click ⚡ to switch between normal and immediate mode
- Visual feedback: priority mode glows amber with border accent
- Submit with Enter key or send button
- Tooltip explains interrupt behavior

### 4. Queue Panel (Frontend)
- Live polling (2s intervals) of queue state
- Visual job cards with progress bars and status icons
- Color-coded status: running (blue), paused (amber), completed (green), failed (red)
- Sections: Now Processing → Priority Stack → Paused → Queue → Recent
- Cancel button per job
- Shows interrupt chain (which job paused which)

### 5. Graph Page Demo Mode Toggle
- Live/Demo toggle button at top of graph visualization
- Demo mode loads the full 3K+ node network into the graph canvas
- Shows node count in toggle button
- Stats bar updates to show demo data source

---

## Files Created (9 new)

| File | Purpose |
|------|---------|
| `backend/app/services/demo/__init__.py` | Module init |
| `backend/app/services/demo/demo_data_generator.py` | 550-line generator with realistic seed data |
| `backend/app/services/queue/__init__.py` | Module init |
| `backend/app/services/queue/priority_queue_service.py` | Priority queue with pause/resume/stack |
| `backend/app/api/routes/demo.py` | Demo mode API (6 endpoints) |
| `backend/app/api/routes/queue.py` | Queue API (4 endpoints + 1 WebSocket) |
| `frontend/src/components/InputBar.tsx` | URL input with priority toggle |
| `frontend/src/components/QueuePanel.tsx` | Live queue visualization |
| `frontend/src/app/queue/page.tsx` | Queue + Demo management page |

## Files Modified (6)

| File | Changes |
|------|---------|
| `backend/app/main.py` | Register demo + queue routers, update root endpoint |
| `frontend/src/lib/types.ts` | Queue types, demo types, job status colors |
| `frontend/src/lib/api.ts` | Demo API + Queue API + Queue WebSocket functions |
| `frontend/src/app/layout.tsx` | Added "Queue" nav link, updated footer version |
| `frontend/src/app/graph-live/page.tsx` | Demo mode toggle, demo data loading |
| `frontend/src/components/graph/GraphControls.tsx` | Added Playlist + Segment to filter |
