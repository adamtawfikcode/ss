"""
Nexum API — Graph Routes

Endpoints for the Social Knowledge Graph:
  - Graph snapshot for visualization
  - Thread traversal
  - Debate detection
  - Entity network
  - User interaction analysis
  - Graph statistics
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.graph_database import graph_db
from app.schemas.schemas import (
    GraphSnapshotSchema,
    GraphStats,
    GraphTraversalRequest,
    ThreadAnalysis,
)
from app.services.graph.graph_service import graph_service
from app.services.graph.traversal_service import traversal_service

router = APIRouter(prefix="/graph", tags=["Graph"])


@router.get("/snapshot")
async def get_graph_snapshot(
    node_types: Optional[str] = Query(None, description="Comma-separated: Video,Comment,Entity,CommentAuthor,Channel"),
    limit: int = Query(500, ge=10, le=1000),
    center_id: Optional[str] = None,
    center_type: Optional[str] = None,
):
    """Get a sampled graph snapshot for live visualization."""
    types = node_types.split(",") if node_types else None
    return await graph_service.get_graph_snapshot(
        node_types=types,
        limit=limit,
        center_node_id=center_id,
        center_node_type=center_type,
    )


@router.get("/stats")
async def get_graph_stats():
    """Get graph-wide statistics."""
    return await graph_db.get_stats()


@router.get("/health")
async def graph_health():
    """Check Neo4j connectivity."""
    return await graph_db.health_check()


# ── Thread Traversal ─────────────────────────────────────────────────────

@router.get("/thread/{comment_id}")
async def get_full_thread(comment_id: str):
    """Retrieve the full thread tree from any comment."""
    return await traversal_service.get_full_thread(comment_id)


@router.get("/thread/{comment_id}/descendants")
async def get_descendants(comment_id: str, max_depth: int = Query(20, ge=1, le=50)):
    """Get all descendants of a comment."""
    return await traversal_service.get_descendants(comment_id, max_depth)


@router.get("/thread/{comment_id}/parents")
async def get_parent_chain(comment_id: str):
    """Get the full parent chain from a comment to root."""
    return await traversal_service.get_parent_chain(comment_id)


@router.get("/thread/{comment_id}/analysis")
async def get_thread_analysis(comment_id: str):
    """Get full thread analysis (depth, authors, sentiment, debate score)."""
    return await traversal_service.get_thread_analysis(comment_id)


# ── Debate & Chain Analysis ──────────────────────────────────────────────

@router.get("/debates/{video_id}")
async def detect_debates(video_id: str, min_depth: int = Query(3, ge=1)):
    """Find debate clusters (high sentiment variance threads) in a video."""
    return await traversal_service.detect_debate_clusters(video_id, min_depth)


@router.get("/chains/{video_id}")
async def find_longest_chains(video_id: str, top_n: int = Query(10, ge=1, le=50)):
    """Find the longest reply chains in a video."""
    return await traversal_service.find_longest_chains(video_id, top_n)


@router.get("/branched/{video_id}")
async def find_most_branched(video_id: str, top_n: int = Query(10, ge=1, le=50)):
    """Find most-branched thread points in a video."""
    return await traversal_service.find_most_branched_threads(video_id, top_n)


# ── User Interaction ─────────────────────────────────────────────────────

@router.get("/users/loops")
async def detect_interaction_loops(min_exchanges: int = Query(3, ge=1)):
    """Find user pairs with bidirectional reply loops."""
    return await traversal_service.detect_user_interaction_loops(min_exchanges)


@router.get("/users/{author_id}/overlap")
async def get_user_overlap(author_id: str):
    """Find users who comment on the same videos."""
    return await traversal_service.get_user_overlap(author_id)


# ── Entity Network ───────────────────────────────────────────────────────

@router.get("/entities/{entity_name}/network")
async def get_entity_network(entity_name: str, depth: int = Query(2, ge=1, le=4)):
    """Get co-occurrence network around an entity."""
    return await traversal_service.get_entity_network(entity_name.lower(), depth)


@router.get("/topics/{topic_name}/timeline")
async def get_topic_timeline(topic_name: str):
    """Get topic mention growth over time."""
    return await traversal_service.get_topic_growth_timeline(topic_name)
