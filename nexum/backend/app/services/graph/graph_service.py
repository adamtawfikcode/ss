"""
Nexum Graph Service — Neo4j Operations

Manages all graph node and edge CRUD operations.
Syncs data from PostgreSQL models into Neo4j for relationship queries.

Node Types: Video, Comment, CommentAuthor, Entity, Topic, Channel, Playlist, Segment
Edge Types: WROTE, REPLIES_TO, ON, MENTIONS, REFERENCES_TIMESTAMP,
            REPLIED_TO, COMMENTED_ON, PARTICIPATED_IN, OVERLAPS_WITH,
            ENGAGED_WITH, UPLOADED, COVERS, SHARES_AUDIENCE_WITH,
            APPEARS_IN, MENTIONED_IN, DISCUSSED_IN, CO_OCCURS_WITH,
            DISCUSSES, CONTAINS, HAS_PLAYLIST
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.core.graph_database import graph_db

logger = logging.getLogger(__name__)
settings = get_settings()


class GraphService:
    """High-level graph operations for the Social Knowledge Graph."""

    # ═══════════════════════════════════════════════════════════════════════
    # Node Creation / Update
    # ═══════════════════════════════════════════════════════════════════════

    async def upsert_video_node(self, video_id: str, data: Dict[str, Any]):
        """Create or update a Video node."""
        await graph_db.execute_write(
            """
            MERGE (v:Video {video_id: $video_id})
            SET v += $data, v.updated_at = datetime()
            """,
            {"video_id": video_id, "data": data},
        )

    async def upsert_comment_node(self, comment_id: str, data: Dict[str, Any]):
        """Create or update a Comment node."""
        await graph_db.execute_write(
            """
            MERGE (c:Comment {comment_id: $comment_id})
            SET c += $data, c.updated_at = datetime()
            """,
            {"comment_id": comment_id, "data": data},
        )

    async def upsert_author_node(self, author_id: str, data: Dict[str, Any]):
        """Create or update a CommentAuthor node."""
        await graph_db.execute_write(
            """
            MERGE (u:CommentAuthor {author_id: $author_id})
            SET u += $data, u.updated_at = datetime()
            """,
            {"author_id": author_id, "data": data},
        )

    async def upsert_entity_node(self, canonical_name: str, data: Dict[str, Any]):
        """Create or update an Entity node."""
        await graph_db.execute_write(
            """
            MERGE (e:Entity {canonical_name: $name})
            SET e += $data, e.updated_at = datetime()
            """,
            {"name": canonical_name, "data": data},
        )

    async def upsert_topic_node(self, topic_name: str, data: Dict[str, Any] = None):
        """Create or update a Topic node."""
        await graph_db.execute_write(
            """
            MERGE (t:Topic {name: $name})
            SET t += $data, t.updated_at = datetime()
            """,
            {"name": topic_name, "data": data or {}},
        )

    async def upsert_channel_node(self, channel_id: str, data: Dict[str, Any]):
        """Create or update a Channel node."""
        await graph_db.execute_write(
            """
            MERGE (ch:Channel {channel_id: $channel_id})
            SET ch += $data, ch.updated_at = datetime()
            """,
            {"channel_id": channel_id, "data": data},
        )

    async def upsert_playlist_node(self, playlist_id: str, data: Dict[str, Any]):
        """Create or update a Playlist node."""
        await graph_db.execute_write(
            """
            MERGE (p:Playlist {playlist_id: $playlist_id})
            SET p += $data, p.updated_at = datetime()
            """,
            {"playlist_id": playlist_id, "data": data},
        )

    async def upsert_segment_node(self, segment_id: str, data: Dict[str, Any]):
        """Create or update a Segment node."""
        await graph_db.execute_write(
            """
            MERGE (s:Segment {segment_id: $segment_id})
            SET s += $data, s.updated_at = datetime()
            """,
            {"segment_id": segment_id, "data": data},
        )

    async def link_playlist_contains(self, playlist_id: str, video_id: str, position: int = 0):
        """PLAYLIST -[:CONTAINS {position}]-> VIDEO"""
        await graph_db.execute_write(
            """
            MATCH (p:Playlist {playlist_id: $pid}), (v:Video {video_id: $vid})
            MERGE (p)-[r:CONTAINS]->(v)
            SET r.position = $pos
            """,
            {"pid": playlist_id, "vid": video_id, "pos": position},
        )

    async def link_channel_has_playlist(self, channel_id: str, playlist_id: str):
        """CHANNEL -[:HAS_PLAYLIST]-> PLAYLIST"""
        await graph_db.execute_write(
            """
            MATCH (ch:Channel {channel_id: $chid}), (p:Playlist {playlist_id: $pid})
            MERGE (ch)-[:HAS_PLAYLIST]->(p)
            """,
            {"chid": channel_id, "pid": playlist_id},
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Edge Creation
    # ═══════════════════════════════════════════════════════════════════════

    async def link_comment_to_video(self, comment_id: str, video_id: str):
        """COMMENT -[:ON]-> VIDEO"""
        await graph_db.execute_write(
            """
            MATCH (c:Comment {comment_id: $cid}), (v:Video {video_id: $vid})
            MERGE (c)-[:ON]->(v)
            """,
            {"cid": comment_id, "vid": video_id},
        )

    async def link_comment_replies_to(self, child_id: str, parent_id: str):
        """COMMENT -[:REPLIES_TO]-> COMMENT"""
        await graph_db.execute_write(
            """
            MATCH (child:Comment {comment_id: $child}), (parent:Comment {comment_id: $parent})
            MERGE (child)-[:REPLIES_TO]->(parent)
            """,
            {"child": child_id, "parent": parent_id},
        )

    async def link_author_wrote(self, author_id: str, comment_id: str):
        """USER -[:WROTE]-> COMMENT"""
        await graph_db.execute_write(
            """
            MATCH (u:CommentAuthor {author_id: $uid}), (c:Comment {comment_id: $cid})
            MERGE (u)-[:WROTE]->(c)
            """,
            {"uid": author_id, "cid": comment_id},
        )

    async def link_author_commented_on(self, author_id: str, video_id: str):
        """USER -[:COMMENTED_ON]-> VIDEO"""
        await graph_db.execute_write(
            """
            MATCH (u:CommentAuthor {author_id: $uid}), (v:Video {video_id: $vid})
            MERGE (u)-[:COMMENTED_ON]->(v)
            """,
            {"uid": author_id, "vid": video_id},
        )

    async def link_author_replied_to(self, author_a: str, author_b: str):
        """USER -[:REPLIED_TO]-> USER (when A replies to B's comment)"""
        if author_a == author_b:
            return
        await graph_db.execute_write(
            """
            MATCH (a:CommentAuthor {author_id: $a}), (b:CommentAuthor {author_id: $b})
            MERGE (a)-[r:REPLIED_TO]->(b)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            {"a": author_a, "b": author_b},
        )

    async def link_comment_mentions_entity(self, comment_id: str, entity_name: str):
        """COMMENT -[:MENTIONS]-> ENTITY"""
        await graph_db.execute_write(
            """
            MATCH (c:Comment {comment_id: $cid}), (e:Entity {canonical_name: $ename})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            {"cid": comment_id, "ename": entity_name},
        )

    async def link_comment_references_segment(self, comment_id: str, segment_id: str, timestamp: float):
        """COMMENT -[:REFERENCES_TIMESTAMP {at}]-> SEGMENT"""
        await graph_db.execute_write(
            """
            MERGE (seg:Segment {segment_id: $sid})
            WITH seg
            MATCH (c:Comment {comment_id: $cid})
            MERGE (c)-[:REFERENCES_TIMESTAMP {at: $ts}]->(seg)
            """,
            {"cid": comment_id, "sid": segment_id, "ts": timestamp},
        )

    async def link_entity_co_occurrence(self, entity_a: str, entity_b: str, source_id: str):
        """ENTITY -[:CO_OCCURS_WITH]-> ENTITY"""
        if entity_a == entity_b:
            return
        await graph_db.execute_write(
            """
            MATCH (a:Entity {canonical_name: $a}), (b:Entity {canonical_name: $b})
            MERGE (a)-[r:CO_OCCURS_WITH]->(b)
            ON CREATE SET r.count = 1, r.sources = [$src]
            ON MATCH SET r.count = r.count + 1
            """,
            {"a": entity_a, "b": entity_b, "src": source_id},
        )

    async def link_channel_uploaded(self, channel_id: str, video_id: str):
        """CHANNEL -[:UPLOADED]-> VIDEO"""
        await graph_db.execute_write(
            """
            MATCH (ch:Channel {channel_id: $chid}), (v:Video {video_id: $vid})
            MERGE (ch)-[:UPLOADED]->(v)
            """,
            {"chid": channel_id, "vid": video_id},
        )

    async def link_channel_covers_topic(self, channel_id: str, topic_name: str):
        """CHANNEL -[:COVERS]-> TOPIC"""
        await graph_db.execute_write(
            """
            MATCH (ch:Channel {channel_id: $chid})
            MERGE (t:Topic {name: $topic})
            MERGE (ch)-[:COVERS]->(t)
            """,
            {"chid": channel_id, "topic": topic_name},
        )

    async def link_author_discusses_entity(self, author_id: str, entity_name: str):
        """USER -[:DISCUSSES]-> ENTITY"""
        await graph_db.execute_write(
            """
            MATCH (u:CommentAuthor {author_id: $uid}), (e:Entity {canonical_name: $ename})
            MERGE (u)-[r:DISCUSSES]->(e)
            ON CREATE SET r.count = 1
            ON MATCH SET r.count = r.count + 1
            """,
            {"uid": author_id, "ename": entity_name},
        )

    async def detect_audience_overlap(self, min_shared_users: int = 3):
        """
        Detect SHARES_AUDIENCE_WITH edges between channels.
        Two channels share audience if N+ users commented on videos from both.
        """
        await graph_db.execute_write(
            """
            MATCH (u:CommentAuthor)-[:COMMENTED_ON]->(v1:Video)<-[:UPLOADED]-(ch1:Channel),
                  (u)-[:COMMENTED_ON]->(v2:Video)<-[:UPLOADED]-(ch2:Channel)
            WHERE ch1 <> ch2
            WITH ch1, ch2, count(DISTINCT u) AS shared
            WHERE shared >= $min
            MERGE (ch1)-[r:SHARES_AUDIENCE_WITH]->(ch2)
            SET r.shared_users = shared
            """,
            {"min": min_shared_users},
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Graph Snapshot for Visualization
    # ═══════════════════════════════════════════════════════════════════════

    async def get_graph_snapshot(
        self,
        node_types: Optional[List[str]] = None,
        limit: int = 500,
        center_node_id: Optional[str] = None,
        center_node_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get a sampled graph snapshot for visualization.
        Returns nodes and edges, limited to max visible count.
        """
        if center_node_id and center_node_type:
            return await self._ego_graph(center_node_id, center_node_type, limit)

        # Default: sample across node types
        allowed = node_types or ["Video", "Comment", "CommentAuthor", "Entity", "Channel", "Playlist"]
        per_type = max(limit // len(allowed), 10)

        nodes = []
        node_ids = set()

        for label in allowed:
            result = await graph_db.execute_read(
                f"""
                MATCH (n:{label})
                RETURN n, labels(n) AS labels
                ORDER BY rand()
                LIMIT $lim
                """,
                {"lim": per_type},
            )
            for record in result:
                n = record["n"]
                nid = self._node_id(n, label)
                if nid not in node_ids:
                    node_ids.add(nid)
                    nodes.append({
                        "id": nid,
                        "label": self._node_label(n, label),
                        "node_type": label,
                        "data": dict(n),
                    })

        # Fetch edges between these nodes
        edges = await self._fetch_edges_for_nodes(node_ids)

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "sampled": True,
        }

    async def _ego_graph(self, node_id: str, node_type: str, limit: int) -> Dict:
        """Get subgraph centered on a specific node."""
        id_field = self._id_field(node_type)
        result = await graph_db.execute_read(
            f"""
            MATCH (center:{node_type} {{{id_field}: $nid}})
            CALL apoc.path.subgraphAll(center, {{maxLevel: 2, limit: $lim}})
            YIELD nodes, relationships
            RETURN nodes, relationships
            """,
            {"nid": node_id, "lim": limit},
        )
        if not result:
            # Fallback without APOC
            return await self._ego_graph_basic(node_id, node_type, limit)
        return self._parse_subgraph(result[0])

    async def _ego_graph_basic(self, node_id: str, node_type: str, limit: int) -> Dict:
        """Fallback ego graph without APOC."""
        id_field = self._id_field(node_type)
        result = await graph_db.execute_read(
            f"""
            MATCH path = (center:{node_type} {{{id_field}: $nid}})-[*1..2]-(neighbor)
            WITH nodes(path) AS ns, relationships(path) AS rs
            LIMIT $lim
            UNWIND ns AS n
            WITH collect(DISTINCT n) AS nodes,
                 collect(DISTINCT rs) AS all_rels
            UNWIND all_rels AS rel_list
            UNWIND rel_list AS r
            RETURN nodes, collect(DISTINCT r) AS rels
            LIMIT 1
            """,
            {"nid": node_id, "lim": limit},
        )
        nodes = []
        edges = []
        if result:
            for n in (result[0].get("nodes") or [])[:limit]:
                labels = list(n.labels) if hasattr(n, "labels") else ["Unknown"]
                lbl = labels[0] if labels else "Unknown"
                nid = self._node_id(n, lbl)
                nodes.append({"id": nid, "label": self._node_label(n, lbl), "node_type": lbl, "data": dict(n)})
            for r in (result[0].get("rels") or []):
                edges.append({"source": str(r.start_node.element_id), "target": str(r.end_node.element_id), "edge_type": r.type, "weight": 1.0})
        return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges), "sampled": True}

    async def _fetch_edges_for_nodes(self, node_ids: set) -> List[Dict]:
        """Fetch all edges between a set of known nodes."""
        if not node_ids:
            return []
        result = await graph_db.execute_read(
            """
            MATCH (a)-[r]->(b)
            WHERE elementId(a) IN $ids OR elementId(b) IN $ids
            RETURN elementId(a) AS src, elementId(b) AS tgt, type(r) AS etype,
                   properties(r) AS props
            LIMIT 2000
            """,
            {"ids": list(node_ids)},
        )
        return [
            {
                "source": r["src"],
                "target": r["tgt"],
                "edge_type": r["etype"],
                "weight": r["props"].get("count", 1) if r["props"] else 1,
            }
            for r in result
        ]

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _id_field(node_type: str) -> str:
        mapping = {
            "Video": "video_id",
            "Comment": "comment_id",
            "CommentAuthor": "author_id",
            "Entity": "canonical_name",
            "Topic": "name",
            "Channel": "channel_id",
            "Playlist": "playlist_id",
            "Segment": "segment_id",
        }
        return mapping.get(node_type, "id")

    @staticmethod
    def _node_id(node, label: str) -> str:
        fields = ["video_id", "comment_id", "author_id", "canonical_name", "name", "channel_id", "playlist_id", "segment_id"]
        for f in fields:
            v = node.get(f)
            if v:
                return str(v)
        return str(node.element_id) if hasattr(node, "element_id") else str(id(node))

    @staticmethod
    def _node_label(node, label: str) -> str:
        for f in ["title", "display_name", "canonical_name", "name", "public_name"]:
            v = node.get(f)
            if v:
                return str(v)[:50]
        return label

    @staticmethod
    def _parse_subgraph(record: Dict) -> Dict:
        nodes = []
        edges = []
        for n in (record.get("nodes") or []):
            labels = list(n.labels) if hasattr(n, "labels") else ["Unknown"]
            nodes.append({"id": str(n.element_id), "label": str(dict(n).get("title", labels[0]))[:50], "node_type": labels[0], "data": dict(n)})
        for r in (record.get("relationships") or []):
            edges.append({"source": str(r.start_node.element_id), "target": str(r.end_node.element_id), "edge_type": r.type, "weight": 1.0})
        return {"nodes": nodes, "edges": edges, "total_nodes": len(nodes), "total_edges": len(edges), "sampled": True}


# Module-level singleton
graph_service = GraphService()
