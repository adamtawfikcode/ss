"""
Nexum Graph Database Layer — Neo4j Driver

Manages Neo4j connections for the Social Knowledge Graph.
Vector DB handles semantic similarity; Graph DB handles relationships.

Schema:
  Nodes:  Video, Comment, CommentAuthor, Entity, Topic, Channel, Playlist, Segment
  Edges:  WROTE, REPLIES_TO, ON, MENTIONS, REFERENCES_TIMESTAMP,
          REPLIED_TO, COMMENTED_ON, PARTICIPATED_IN, OVERLAPS_WITH,
          ENGAGED_WITH, UPLOADED, COVERS, SHARES_AUDIENCE_WITH,
          APPEARS_IN, MENTIONED_IN, DISCUSSED_IN, CO_OCCURS_WITH,
          DISCUSSES, CONTAINS, HAS_PLAYLIST
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from neo4j import AsyncDriver, AsyncGraphDatabase, AsyncSession
from neo4j.exceptions import ServiceUnavailable

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class GraphDatabase:
    """Neo4j async driver wrapper with schema bootstrap and query helpers."""

    def __init__(self):
        self._driver: Optional[AsyncDriver] = None

    async def connect(self):
        """Initialize the Neo4j async driver."""
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=settings.neo4j_max_connection_pool_size,
            connection_timeout=30,
        )
        try:
            await self._driver.verify_connectivity()
            logger.info("Neo4j connected", extra={"uri": settings.neo4j_uri})
        except ServiceUnavailable:
            logger.error("Neo4j not reachable — graph features degraded")

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def session(self):
        """Yield an async Neo4j session."""
        if not self._driver:
            await self.connect()
        async with self._driver.session(database=settings.neo4j_database) as session:
            yield session

    # ── Schema Bootstrap ─────────────────────────────────────────────────

    async def ensure_schema(self):
        """Create constraints and indices for all node types."""
        constraints = [
            # Uniqueness constraints (also create index)
            "CREATE CONSTRAINT IF NOT EXISTS FOR (v:Video) REQUIRE v.video_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Comment) REQUIRE c.comment_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:CommentAuthor) REQUIRE u.author_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (ch:Channel) REQUIRE ch.channel_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Playlist) REQUIRE p.playlist_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
        ]
        indices = [
            # Composite / range indices for traversal performance
            "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.video_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.root_thread_id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.depth_level)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Comment) ON (c.timestamp_posted)",
            "CREATE INDEX IF NOT EXISTS FOR (u:CommentAuthor) ON (u.comment_count)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.mention_count)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
        ]
        async with self.session() as session:
            for stmt in constraints + indices:
                try:
                    await session.run(stmt)
                except Exception as e:
                    logger.warning(f"Schema stmt skipped: {e}")
        logger.info("Neo4j schema ensured")

    # ── Generic Query Helpers ────────────────────────────────────────────

    async def execute_write(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute a write transaction."""
        async with self.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) async for record in result]

    async def execute_read(self, query: str, params: Dict[str, Any] = None) -> List[Dict]:
        """Execute a read transaction."""
        async with self.session() as session:
            result = await session.run(query, params or {})
            return [dict(record) async for record in result]

    async def execute_read_single(self, query: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Execute a read and return a single result or None."""
        results = await self.execute_read(query, params)
        return results[0] if results else None

    # ── Health ───────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        try:
            async with self.session() as session:
                result = await session.run("RETURN 1 AS ok")
                record = await result.single()
                return {"status": "healthy", "neo4j": "connected"}
        except Exception as e:
            return {"status": "degraded", "neo4j": str(e)}

    async def get_stats(self) -> Dict[str, int]:
        """Graph-wide node and relationship counts."""
        stats = {}
        labels = ["Video", "Comment", "CommentAuthor", "Entity", "Topic", "Channel", "Playlist", "Segment"]
        async with self.session() as session:
            for label in labels:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                record = await result.single()
                stats[f"{label.lower()}_count"] = record["cnt"] if record else 0
            # Total edges
            result = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            record = await result.single()
            stats["total_edges"] = record["cnt"] if record else 0
        return stats


# Module-level singleton
graph_db = GraphDatabase()
