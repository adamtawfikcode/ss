"""
Nexum Event Streaming — Real-time graph events via WebSocket + SSE.

Events are emitted by backend services whenever the graph mutates
(new node, new edge) and broadcast to all connected graph-live clients.

Node Events:  VIDEO_ADDED, COMMENT_ADDED, USER_CREATED, ENTITY_DISCOVERED, CHANNEL_LINKED
Edge Events:  COMMENT_LINKED, USER_CONNECTED, ENTITY_CO_OCCURRENCE, THREAD_EXPANDED
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from fastapi import WebSocket, WebSocketDisconnect

from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ═══════════════════════════════════════════════════════════════════════════
# Event Types
# ═══════════════════════════════════════════════════════════════════════════

class GraphEventType(str, Enum):
    # Node events
    VIDEO_ADDED = "VIDEO_ADDED"
    COMMENT_ADDED = "COMMENT_ADDED"
    USER_CREATED = "USER_CREATED"
    ENTITY_DISCOVERED = "ENTITY_DISCOVERED"
    CHANNEL_LINKED = "CHANNEL_LINKED"
    # Edge events
    COMMENT_LINKED = "COMMENT_LINKED"
    USER_CONNECTED = "USER_CONNECTED"
    ENTITY_CO_OCCURRENCE = "ENTITY_CO_OCCURRENCE"
    THREAD_EXPANDED = "THREAD_EXPANDED"
    # Meta
    HEARTBEAT = "HEARTBEAT"
    STATS_UPDATE = "STATS_UPDATE"


@dataclass
class GraphEvent:
    event_type: str
    timestamp: float = field(default_factory=time.time)
    node_type: Optional[str] = None       # Video | Comment | CommentAuthor | Entity | Topic | Channel
    node_id: Optional[str] = None
    edge_type: Optional[str] = None       # WROTE | REPLIES_TO | ON | etc.
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None  # Extra payload (label, coords, etc.)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


# ═══════════════════════════════════════════════════════════════════════════
# Connection Manager
# ═══════════════════════════════════════════════════════════════════════════

class GraphEventHub:
    """
    Central hub for real-time graph event broadcasting.

    - Maintains set of active WebSocket connections
    - Keeps a bounded event buffer for replay (last N events)
    - Supports topic filtering per client
    """

    def __init__(self, buffer_size: int = 1000):
        self._connections: Set[WebSocket] = set()
        self._buffer: deque[GraphEvent] = deque(maxlen=buffer_size)
        self._lock = asyncio.Lock()
        self._stats = {
            "total_events_emitted": 0,
            "active_connections": 0,
        }

    # ── Connection Lifecycle ─────────────────────────────────────────────

    async def connect(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self._connections.add(ws)
            self._stats["active_connections"] = len(self._connections)
        logger.info(f"Graph WS connected (total={len(self._connections)})")

    async def disconnect(self, ws: WebSocket):
        async with self._lock:
            self._connections.discard(ws)
            self._stats["active_connections"] = len(self._connections)
        logger.info(f"Graph WS disconnected (total={len(self._connections)})")

    # ── Event Emission ───────────────────────────────────────────────────

    async def emit(self, event: GraphEvent):
        """Emit an event to all connected clients and buffer it."""
        self._buffer.append(event)
        self._stats["total_events_emitted"] += 1

        if not self._connections:
            return

        payload = event.to_json()
        dead: List[WebSocket] = []

        for ws in list(self._connections):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.discard(ws)
                self._stats["active_connections"] = len(self._connections)

    # ── Convenience Emitters ─────────────────────────────────────────────

    async def emit_node(
        self,
        event_type: GraphEventType,
        node_type: str,
        node_id: str,
        data: Optional[Dict] = None,
    ):
        await self.emit(GraphEvent(
            event_type=event_type.value,
            node_type=node_type,
            node_id=node_id,
            data=data,
        ))

    async def emit_edge(
        self,
        event_type: GraphEventType,
        edge_type: str,
        source_id: str,
        target_id: str,
        data: Optional[Dict] = None,
    ):
        await self.emit(GraphEvent(
            event_type=event_type.value,
            edge_type=edge_type,
            source_id=source_id,
            target_id=target_id,
            data=data,
        ))

    # ── Replay ───────────────────────────────────────────────────────────

    async def replay(
        self,
        ws: WebSocket,
        since: Optional[float] = None,
        event_types: Optional[List[str]] = None,
        limit: int = 500,
    ):
        """Send buffered events to a newly connected client for catchup."""
        events = list(self._buffer)
        if since:
            events = [e for e in events if e.timestamp >= since]
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        events = events[-limit:]

        for event in events:
            try:
                await ws.send_text(event.to_json())
            except Exception:
                break

    # ── SSE Fallback Generator ───────────────────────────────────────────

    async def sse_stream(self, since: Optional[float] = None):
        """Async generator for Server-Sent Events fallback."""
        # Replay buffer first
        events = list(self._buffer)
        if since:
            events = [e for e in events if e.timestamp >= since]
        for event in events[-200:]:
            yield f"data: {event.to_json()}\n\n"

        # Then stream live events via polling the buffer tail
        last_ts = events[-1].timestamp if events else time.time()
        while True:
            await asyncio.sleep(0.5)
            new_events = [e for e in self._buffer if e.timestamp > last_ts]
            for event in new_events:
                yield f"data: {event.to_json()}\n\n"
                last_ts = event.timestamp

            # Heartbeat every 15s to keep connection alive
            if not new_events:
                yield f"data: {GraphEvent(event_type='HEARTBEAT').to_json()}\n\n"

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "buffer_size": len(self._buffer),
            "buffer_capacity": self._buffer.maxlen,
        }


# Module-level singleton
event_hub = GraphEventHub(buffer_size=settings.graph_event_buffer_size)
