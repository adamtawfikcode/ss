"""
Nexum Graph Traversal Service

Specialized traversal queries for the threaded conversation graph:
  - Full thread retrieval from any comment
  - All descendants / parent chain
  - Debate cluster detection
  - Longest chain identification
  - Most branched threads
  - User interaction loop detection
  - Entity co-occurrence paths
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from app.core.graph_database import graph_db

logger = logging.getLogger(__name__)


class GraphTraversalService:
    """Specialized Neo4j traversal queries for knowledge graph analysis."""

    # ═══════════════════════════════════════════════════════════════════════
    # Thread Traversal
    # ═══════════════════════════════════════════════════════════════════════

    async def get_full_thread(self, comment_id: str) -> Dict[str, Any]:
        """
        Retrieve full thread tree from any comment.
        Returns the root and all descendants with structure info.
        """
        # First find the root
        root = await graph_db.execute_read_single(
            """
            MATCH (c:Comment {comment_id: $cid})
            OPTIONAL MATCH path = (c)-[:REPLIES_TO*]->(root:Comment)
            WHERE NOT (root)-[:REPLIES_TO]->()
            WITH CASE WHEN root IS NOT NULL THEN root ELSE c END AS root_node
            RETURN root_node.comment_id AS root_id
            """,
            {"cid": comment_id},
        )
        if not root:
            return {"root_id": comment_id, "comments": [], "total_depth": 0}

        root_id = root["root_id"]

        # Get all descendants
        result = await graph_db.execute_read(
            """
            MATCH (root:Comment {comment_id: $rid})
            OPTIONAL MATCH path = (descendant:Comment)-[:REPLIES_TO*]->(root)
            WITH root, collect(DISTINCT descendant) + [root] AS all_comments
            UNWIND all_comments AS c
            OPTIONAL MATCH (author:CommentAuthor)-[:WROTE]->(c)
            RETURN c.comment_id AS id,
                   c.text AS text,
                   c.depth_level AS depth,
                   c.sentiment_score AS sentiment,
                   c.like_count AS likes,
                   c.timestamp_posted AS posted,
                   author.display_name AS author,
                   author.author_id AS author_id
            ORDER BY c.depth_level, c.timestamp_posted
            """,
            {"rid": root_id},
        )

        max_depth = max((r.get("depth", 0) or 0) for r in result) if result else 0

        return {
            "root_id": root_id,
            "comments": result,
            "total_comments": len(result),
            "total_depth": max_depth,
        }

    async def get_descendants(self, comment_id: str, max_depth: int = 20) -> List[Dict]:
        """Retrieve all descendants of a comment."""
        result = await graph_db.execute_read(
            """
            MATCH (parent:Comment {comment_id: $cid})
            MATCH path = (child:Comment)-[:REPLIES_TO*1..]->(parent)
            WHERE length(path) <= $max_depth
            RETURN child.comment_id AS id,
                   child.text AS text,
                   child.depth_level AS depth,
                   child.sentiment_score AS sentiment,
                   length(path) AS distance
            ORDER BY length(path), child.timestamp_posted
            """,
            {"cid": comment_id, "max_depth": max_depth},
        )
        return result

    async def get_parent_chain(self, comment_id: str) -> List[Dict]:
        """Retrieve the full parent chain from a comment up to the root."""
        result = await graph_db.execute_read(
            """
            MATCH (c:Comment {comment_id: $cid})
            MATCH path = (c)-[:REPLIES_TO*]->(ancestor:Comment)
            WITH ancestor, length(path) AS dist
            OPTIONAL MATCH (author:CommentAuthor)-[:WROTE]->(ancestor)
            RETURN ancestor.comment_id AS id,
                   ancestor.text AS text,
                   ancestor.depth_level AS depth,
                   dist AS distance,
                   author.display_name AS author
            ORDER BY dist
            """,
            {"cid": comment_id},
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Thread Analysis
    # ═══════════════════════════════════════════════════════════════════════

    async def detect_debate_clusters(self, video_id: str, min_depth: int = 3) -> List[Dict]:
        """
        Find debate clusters: threads with high sentiment variance.
        A debate has mixed positive/negative sentiments among replies.
        """
        result = await graph_db.execute_read(
            """
            MATCH (root:Comment {video_id: $vid})
            WHERE NOT (root)-[:REPLIES_TO]->()
            MATCH (reply:Comment)-[:REPLIES_TO*]->(root)
            WITH root,
                 count(reply) AS reply_count,
                 avg(reply.sentiment_score) AS avg_sentiment,
                 stDev(reply.sentiment_score) AS sentiment_variance,
                 max(reply.depth_level) AS max_depth
            WHERE reply_count >= 3 AND max_depth >= $min_depth
            RETURN root.comment_id AS root_id,
                   root.text AS root_text,
                   reply_count,
                   round(avg_sentiment * 100) / 100 AS avg_sentiment,
                   round(sentiment_variance * 100) / 100 AS debate_score,
                   max_depth
            ORDER BY sentiment_variance DESC
            LIMIT 20
            """,
            {"vid": video_id, "min_depth": min_depth},
        )
        return result

    async def find_longest_chains(self, video_id: str, top_n: int = 10) -> List[Dict]:
        """Find the longest reply chains in a video's comments."""
        result = await graph_db.execute_read(
            """
            MATCH (leaf:Comment {video_id: $vid})
            WHERE NOT (:Comment)-[:REPLIES_TO]->(leaf)
                  AND (leaf)-[:REPLIES_TO]->()
            MATCH path = (leaf)-[:REPLIES_TO*]->(root:Comment)
            WHERE NOT (root)-[:REPLIES_TO]->()
            WITH root, leaf, length(path) AS chain_length
            ORDER BY chain_length DESC
            LIMIT $n
            RETURN root.comment_id AS root_id,
                   leaf.comment_id AS leaf_id,
                   chain_length
            """,
            {"vid": video_id, "n": top_n},
        )
        return result

    async def find_most_branched_threads(self, video_id: str, top_n: int = 10) -> List[Dict]:
        """Find threads with the most branching (replies to the same comment)."""
        result = await graph_db.execute_read(
            """
            MATCH (parent:Comment {video_id: $vid})<-[:REPLIES_TO]-(child:Comment)
            WITH parent, count(child) AS branch_count
            ORDER BY branch_count DESC
            LIMIT $n
            RETURN parent.comment_id AS comment_id,
                   parent.text AS text,
                   parent.depth_level AS depth,
                   branch_count
            """,
            {"vid": video_id, "n": top_n},
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # User Interaction Analysis
    # ═══════════════════════════════════════════════════════════════════════

    async def detect_user_interaction_loops(self, min_exchanges: int = 3) -> List[Dict]:
        """
        Detect users who repeatedly reply to each other (interaction loops).
        Returns pairs with bidirectional reply counts.
        """
        result = await graph_db.execute_read(
            """
            MATCH (a:CommentAuthor)-[r1:REPLIED_TO]->(b:CommentAuthor),
                  (b)-[r2:REPLIED_TO]->(a)
            WHERE r1.count >= $min AND r2.count >= $min
            RETURN a.author_id AS user_a,
                   a.display_name AS name_a,
                   b.author_id AS user_b,
                   b.display_name AS name_b,
                   r1.count AS a_to_b,
                   r2.count AS b_to_a,
                   r1.count + r2.count AS total_exchanges
            ORDER BY total_exchanges DESC
            LIMIT 50
            """,
            {"min": min_exchanges},
        )
        return result

    async def get_user_overlap(self, author_id: str) -> List[Dict]:
        """Find other users who comment on the same videos."""
        result = await graph_db.execute_read(
            """
            MATCH (u:CommentAuthor {author_id: $uid})-[:COMMENTED_ON]->(v:Video)<-[:COMMENTED_ON]-(other:CommentAuthor)
            WHERE other <> u
            WITH other, count(DISTINCT v) AS shared_videos
            WHERE shared_videos >= 2
            RETURN other.author_id AS author_id,
                   other.display_name AS display_name,
                   shared_videos
            ORDER BY shared_videos DESC
            LIMIT 20
            """,
            {"uid": author_id},
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Entity Graph
    # ═══════════════════════════════════════════════════════════════════════

    async def get_entity_network(self, entity_name: str, depth: int = 2) -> Dict:
        """Get entity co-occurrence network."""
        result = await graph_db.execute_read(
            """
            MATCH (e:Entity {canonical_name: $name})
            MATCH (e)-[r:CO_OCCURS_WITH*1..]->(related:Entity)
            WHERE length(r) <= $depth
            WITH collect(DISTINCT related) + [e] AS entities
            UNWIND entities AS ent
            OPTIONAL MATCH (ent)-[co:CO_OCCURS_WITH]-(other)
            WHERE other IN entities
            RETURN ent.canonical_name AS name,
                   ent.entity_type AS type,
                   ent.mention_count AS mentions,
                   collect(DISTINCT {target: other.canonical_name, count: co.count}) AS connections
            """,
            {"name": entity_name, "depth": depth},
        )
        return {"entity": entity_name, "network": result}

    async def get_topic_growth_timeline(self, topic_name: str) -> List[Dict]:
        """Track when a topic appeared in comments over time."""
        result = await graph_db.execute_read(
            """
            MATCH (c:Comment)-[:MENTIONS]->(e:Entity)
            WHERE e.canonical_name CONTAINS $topic
            WITH date(c.timestamp_posted) AS day, count(c) AS mentions
            RETURN day, mentions
            ORDER BY day
            """,
            {"topic": topic_name.lower()},
        )
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # Aggregate Thread Stats
    # ═══════════════════════════════════════════════════════════════════════

    async def get_thread_analysis(self, root_comment_id: str) -> Dict[str, Any]:
        """Full analysis of a comment thread."""
        result = await graph_db.execute_read_single(
            """
            MATCH (root:Comment {comment_id: $rid})
            OPTIONAL MATCH (reply:Comment)-[:REPLIES_TO*]->(root)
            OPTIONAL MATCH (author:CommentAuthor)-[:WROTE]->(reply)
            WITH root,
                 collect(DISTINCT reply) AS replies,
                 collect(DISTINCT author) AS authors
            RETURN root.comment_id AS root_id,
                   size(replies) + 1 AS total_comments,
                   size(authors) AS unique_authors,
                   CASE WHEN size(replies) > 0
                     THEN reduce(s = 0.0, r IN replies | s + coalesce(r.sentiment_score, 0)) / size(replies)
                     ELSE 0.0
                   END AS avg_sentiment,
                   CASE WHEN size(replies) > 0
                     THEN max([r IN replies | r.depth_level])
                     ELSE 0
                   END AS max_depth
            """,
            {"rid": root_comment_id},
        )
        if not result:
            return {}

        return {
            "root_comment_id": result["root_id"],
            "total_comments": result["total_comments"],
            "unique_authors": result["unique_authors"],
            "avg_sentiment": round(result["avg_sentiment"], 3),
            "total_depth": result["max_depth"] or 0,
        }


# Module-level singleton
traversal_service = GraphTraversalService()
