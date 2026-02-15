"""
Nexum API — WebSocket & SSE Routes for Live Graph Visualization

WebSocket primary, SSE fallback. Supports:
  - Real-time graph event streaming
  - Replay from timestamp
  - Event type filtering
  - Connection lifecycle management
"""
from __future__ import annotations

import json
import logging
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from app.core.events import event_hub

logger = logging.getLogger(__name__)
router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/graph")
async def graph_websocket(
    ws: WebSocket,
    replay_since: Optional[float] = Query(None),
    event_types: Optional[str] = Query(None),
):
    """
    WebSocket endpoint for live graph event streaming.

    Query params:
      replay_since: Unix timestamp to replay buffered events from
      event_types: Comma-separated event type filter
    """
    await event_hub.connect(ws)

    try:
        # Replay buffered events for catchup
        types_filter = event_types.split(",") if event_types else None
        await event_hub.replay(ws, since=replay_since, event_types=types_filter)

        # Listen for client messages (filters, ping, etc.)
        while True:
            try:
                data = await ws.receive_text()
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await ws.send_text(json.dumps({"type": "pong"}))
                elif msg.get("type") == "filter":
                    # Client-side filtering handled in the frontend
                    pass
                elif msg.get("type") == "replay":
                    since = msg.get("since")
                    await event_hub.replay(ws, since=since)

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await event_hub.disconnect(ws)


@router.get("/sse/graph")
async def graph_sse(
    replay_since: Optional[float] = Query(None),
):
    """
    SSE fallback endpoint for browsers that don't support WebSocket.
    """
    return StreamingResponse(
        event_hub.sse_stream(since=replay_since),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/ws/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return event_hub.get_stats()
