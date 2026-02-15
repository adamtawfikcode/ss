"""
Nexum API — Demo Mode Routes (v4.2).

Demo environment with realistic mock data + deep analytics signals:
  - GET  /demo/snapshot     — Full graph snapshot (all node types, big network)
  - GET  /demo/stats        — Demo dataset statistics (incl. signal counts)
  - POST /demo/nodes        — Create a new demo node
  - GET  /demo/nodes/{type} — List nodes of a specific type
  - POST /demo/reset        — Regenerate fresh demo data
  - GET  /demo/status       — Demo mode on/off status
  - GET  /demo/signals/{signal_type} — List signal records by type
  - GET  /demo/signals/stats — Aggregate signal statistics
  - POST /demo/recrawl/{video_id} — Simulate recrawl of a video (detect changes)
  - GET  /demo/stale        — Find nodes that need recrawling (oldest updated_at)
"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.services.demo.demo_data_generator import DemoDataGenerator

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/demo", tags=["Demo"])

# Module-level demo state
_generator = DemoDataGenerator(seed=42)
_demo_active = True

VALID_NODE_TYPES = ["Video", "Channel", "Comment", "CommentAuthor", "Entity", "Topic", "Playlist", "Segment"]

VALID_SIGNAL_TYPES = [
    "recrawl_events", "cursor_heatmaps", "text_edit_events", "face_camera_ratios",
    "background_changes", "breathing_patterns", "ambient_bleeds", "room_fingerprints",
    "laughter_events", "upload_patterns", "title_edit_history", "thumbnail_analyses",
    "link_health", "pinned_comment_history", "sentiment_drifts", "deleted_shadows",
    "broll_reuses", "sponsor_segments", "intro_outro_evolutions", "linguistic_profiles",
    "commenter_migrations", "entity_propagations", "parasocial_indices", "topic_gestations",
]


class CreateNodeRequest(BaseModel):
    node_type: str = Field(..., description="One of: Video, Channel, Comment, CommentAuthor, Entity, Topic, Playlist, Segment")
    data: dict = Field(default_factory=dict, description="Optional overrides for auto-generated fields")


class CreateNodeResponse(BaseModel):
    id: str
    label: str
    node_type: str
    data: dict
    edges_created: int = 0


# ── Core Endpoints ───────────────────────────────────────────────────

@router.get("/status")
async def demo_status():
    return {"demo_active": _demo_active, "generated": _generator._generated}


@router.post("/activate")
async def activate_demo(active: bool = True):
    global _demo_active
    _demo_active = active
    if active and not _generator._generated:
        _generator.generate()
    return {"demo_active": _demo_active}


@router.get("/snapshot")
async def get_demo_snapshot(
    node_types: Optional[str] = Query(None, description="Comma-separated filter: Video,Channel,Entity,..."),
    limit: int = Query(5000, ge=10, le=10000),
):
    """Full demo graph snapshot — thousands of interconnected nodes."""
    data = _generator.generate()

    if node_types:
        types = set(t.strip() for t in node_types.split(","))
        kept_ids = set()
        nodes = []
        for n in data["nodes"]:
            if n["node_type"] in types:
                nodes.append(n)
                kept_ids.add(n["id"])
        edges = [e for e in data["edges"] if e["source"] in kept_ids and e["target"] in kept_ids]
    else:
        nodes = data["nodes"][:limit]
        node_ids = set(n["id"] for n in nodes)
        edges = [e for e in data["edges"] if e["source"] in node_ids and e["target"] in node_ids]

    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "sampled": len(nodes) < data["total_nodes"],
        "stats": data["stats"],
        "signal_stats": data.get("signal_stats", {}),
        "demo_mode": True,
    }


@router.get("/stats")
async def get_demo_stats():
    """Dataset statistics including signal counts."""
    data = _generator.generate()
    stats = data["stats"]

    edge_types: dict = {}
    for e in data["edges"]:
        edge_types[e["edge_type"]] = edge_types.get(e["edge_type"], 0) + 1

    return {
        "node_counts": {
            "Channel": stats["channels"],
            "Video": stats["videos"],
            "Comment": stats["comments"],
            "CommentAuthor": stats["authors"],
            "Entity": stats["entities"],
            "Topic": stats["topics"],
            "Playlist": stats["playlists"],
            "Segment": stats["segments"],
        },
        "total_nodes": data["total_nodes"],
        "total_edges": data["total_edges"],
        "edge_type_distribution": edge_types,
        "signal_stats": data.get("signal_stats", {}),
        "demo_mode": True,
    }


@router.post("/nodes", response_model=CreateNodeResponse)
async def create_demo_node(req: CreateNodeRequest):
    """Create a new demo node with auto-generated realistic data + updated_at."""
    if req.node_type not in VALID_NODE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid node_type '{req.node_type}'. Must be one of: {', '.join(VALID_NODE_TYPES)}")

    _generator.generate()
    edge_count_before = len(_generator.edges)
    node = _generator.create_node(req.node_type, req.data)
    edges_created = len(_generator.edges) - edge_count_before

    return CreateNodeResponse(
        id=node["id"], label=node["label"], node_type=node["node_type"],
        data=node["data"], edges_created=edges_created,
    )


@router.get("/nodes/{node_type}")
async def list_demo_nodes(
    node_type: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """List demo nodes with pagination."""
    if node_type not in VALID_NODE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid node_type. Must be one of: {', '.join(VALID_NODE_TYPES)}")

    _generator.generate()
    type_map = {
        "Channel": _generator.channels, "Video": _generator.videos,
        "Comment": _generator.comments, "CommentAuthor": _generator.authors,
        "Entity": _generator.entities, "Topic": _generator.topics,
        "Playlist": _generator.playlists, "Segment": _generator.segments,
    }
    items = type_map.get(node_type, [])
    start = (page - 1) * page_size
    return {"node_type": node_type, "total": len(items), "page": page, "page_size": page_size, "items": items[start:start + page_size]}


@router.post("/reset")
async def reset_demo():
    """Regenerate demo data from scratch."""
    global _generator
    _generator = DemoDataGenerator(seed=int(__import__("time").time()) % 10000)
    data = _generator.generate()
    return {"status": "regenerated", "stats": data["stats"], "signal_stats": data.get("signal_stats", {})}


# ── Signal Data Endpoints ────────────────────────────────────────────

@router.get("/signals/types")
async def list_signal_types():
    """List all available signal types with counts."""
    _generator.generate()
    sig = _generator.signals
    if not sig:
        return {"signal_types": [], "total_types": 0}
    return {"signal_types": VALID_SIGNAL_TYPES, "total_types": len(VALID_SIGNAL_TYPES), "counts": sig.stats()}


@router.get("/signals/stats")
async def get_signal_stats():
    """Aggregate signal statistics across all categories."""
    _generator.generate()
    sig = _generator.signals
    if not sig:
        return {"error": "Signals not generated"}

    counts = sig.stats()
    total = sum(counts.values())
    return {
        "total_signal_records": total,
        "by_type": counts,
        "categories": {
            "temporal_visual": sum(counts.get(k, 0) for k in ["cursor_heatmaps", "text_edit_events", "face_camera_ratios", "background_changes"]),
            "audio_micro": sum(counts.get(k, 0) for k in ["breathing_patterns", "ambient_bleeds", "room_fingerprints", "laughter_events"]),
            "behavioral": sum(counts.get(k, 0) for k in ["upload_patterns", "title_edit_history", "thumbnail_analyses", "link_health"]),
            "comment_archaeology": sum(counts.get(k, 0) for k in ["pinned_comment_history", "sentiment_drifts", "deleted_shadows"]),
            "cross_video_forensics": sum(counts.get(k, 0) for k in ["broll_reuses", "sponsor_segments", "intro_outro_evolutions"]),
            "linguistic": counts.get("linguistic_profiles", 0),
            "graph_relational": sum(counts.get(k, 0) for k in ["commenter_migrations", "entity_propagations", "parasocial_indices", "topic_gestations"]),
            "recrawl": counts.get("recrawl_events", 0),
        },
    }


@router.get("/signals/{signal_type}")
async def get_signal_data(
    signal_type: str,
    video_id: Optional[str] = Query(None, description="Filter by video ID"),
    channel_id: Optional[str] = Query(None, description="Filter by channel ID"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
):
    """Retrieve signal records by type, with optional video/channel filter."""
    if signal_type not in VALID_SIGNAL_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid signal type. Must be one of: {', '.join(VALID_SIGNAL_TYPES)}")

    _generator.generate()
    sig = _generator.signals
    if not sig:
        raise HTTPException(status_code=503, detail="Signals not generated yet")

    items = getattr(sig, signal_type, [])
    if video_id:
        items = [r for r in items if r.get("video_id") == video_id]
    if channel_id:
        items = [r for r in items if r.get("channel_id") == channel_id]

    total = len(items)
    start = (page - 1) * page_size
    return {"signal_type": signal_type, "total": total, "page": page, "page_size": page_size, "items": items[start:start + page_size]}


# ── Recrawl Simulation ──────────────────────────────────────────────

@router.post("/recrawl/{video_id}")
async def simulate_recrawl(video_id: str):
    """
    Simulate recrawling a video — detect what changed since last crawl.

    Generates a delta showing updated fields (view_count, comment_count,
    description changes, new/deleted comments, etc.) and updates the
    video's updated_at timestamp.
    """
    _generator.generate()
    vid = next((v for v in _generator.videos if v["id"] == video_id), None)
    if not vid:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found in demo")

    now = datetime.now(timezone.utc).isoformat()
    old_updated = vid.get("updated_at", now)

    changes = {}
    if random.random() < 0.9:
        old_v = vid.get("view_count", 1000)
        new_v = old_v + random.randint(50, 20000)
        changes["view_count"] = [old_v, new_v]
        vid["view_count"] = new_v
    if random.random() < 0.7:
        old_lc = vid.get("like_count", 100)
        new_lc = old_lc + random.randint(5, 2000)
        changes["like_count"] = [old_lc, new_lc]
        vid["like_count"] = new_lc
    if random.random() < 0.4:
        old_cc = vid.get("comment_count", 10)
        changes["comment_count"] = [old_cc, old_cc + random.randint(1, 200)]
    desc_changed = random.random() < 0.1
    title_changed = random.random() < 0.05
    if desc_changed:
        changes["description"] = ["[old description]", "[updated description with new links]"]
    if title_changed:
        changes["title"] = [vid.get("title", ""), vid.get("title", "") + " [Updated]"]
        vid["title"] = changes["title"][1]

    new_comments = random.randint(0, 50)
    deleted_comments = random.randint(0, 8)

    vid["updated_at"] = now
    vid["last_crawled_at"] = now

    recrawl_event = {
        "id": str(random.getrandbits(128)),
        "video_id": video_id,
        "trigger": "manual",
        "fields_changed": list(changes.keys()),
        "delta_json": changes,
        "comments_added": new_comments,
        "comments_deleted": deleted_comments,
        "description_changed": desc_changed,
        "title_changed": title_changed,
        "duration_ms": random.randint(500, 8000),
        "created_at": now,
        "updated_at": now,
    }

    sig = _generator.signals
    if sig:
        sig.recrawl_events.append(recrawl_event)

    return {
        "video_id": video_id,
        "previous_updated_at": old_updated,
        "new_updated_at": now,
        "fields_changed": list(changes.keys()),
        "delta": changes,
        "comments_added": new_comments,
        "comments_deleted": deleted_comments,
        "recrawl_event": recrawl_event,
    }


# ── Staleness Detection ─────────────────────────────────────────────

@router.get("/stale")
async def get_stale_nodes(
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    max_age_hours: int = Query(168, ge=1, description="Nodes older than this (hours). Default: 168 = 7 days"),
    limit: int = Query(50, ge=1, le=500),
):
    """
    Find nodes that need recrawling — sorted by oldest updated_at.
    """
    _generator.generate()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()

    type_map = {
        "Channel": _generator.channels, "Video": _generator.videos,
        "Comment": _generator.comments, "CommentAuthor": _generator.authors,
        "Entity": _generator.entities, "Topic": _generator.topics,
        "Playlist": _generator.playlists, "Segment": _generator.segments,
    }
    if node_type and node_type not in type_map:
        raise HTTPException(status_code=400, detail=f"Invalid node_type. Must be one of: {', '.join(type_map.keys())}")

    stale_nodes = []
    sources = {node_type: type_map[node_type]} if node_type else type_map

    for ntype, items in sources.items():
        for item in items:
            ua = item.get("updated_at") or item.get("created_at") or ""
            if ua < cutoff:
                stale_nodes.append({
                    "id": item.get("id"),
                    "node_type": ntype,
                    "label": item.get("name") or item.get("title") or item.get("display_name") or item.get("canonical_name") or item.get("text", "")[:50],
                    "updated_at": ua,
                    "staleness_hours": _hours_since(ua),
                })

    stale_nodes.sort(key=lambda x: x.get("updated_at", ""))
    return {"cutoff_hours": max_age_hours, "total_stale": len(stale_nodes), "items": stale_nodes[:limit]}


def _hours_since(iso_str: str) -> int:
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int((datetime.now(timezone.utc) - dt).total_seconds() / 3600)
    except (ValueError, TypeError):
        return -1
