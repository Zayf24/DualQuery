"""
DualQuery Cost-Based Physical Planner.
Chooses and compiles:
- Plan A: High-Selectivity Pushdown (Relational Filter -> Exact Cosine Scan)
- Plan B: Low-Selectivity (HNSW Index + BM25, fused with RRF -> Relational Filter)
"""

import math
import re
from typing import List, Optional, Any
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.engine.cardinality_probe import CardinalityProbe, CardinalityEstimate
from dualquery.engine.sql_normalizer import get_required_joins


class HybridQueryRequest(BaseModel):
    """Specification of a compound hybrid query."""
    sql_filter: str = Field(default="1=1", description="Relational SQL WHERE clause")
    semantic_query: str = Field(description="Natural language complaint/inquiry text for vector search")
    bm25_keywords: Optional[str] = Field(default=None, description="Keywords for lexical BM25 matching (auto-extracted if None)")
    aggregations: List[str] = Field(
        default_factory=lambda: [
            "COUNT(DISTINCT orders.order_id) AS total_orders",
            "ROUND(SUM(order_payments.payment_value), 2) AS total_amount"
        ],
        description="Target OLAP aggregations, one table family per metric"
    )
    limit: int = Field(default=5, ge=1, le=50, description="Number of top cited review evidence records to return")
    match_top_percent: float = Field(
        default=1.0, gt=0.0, le=100.0,
        description="A review matches the semantic query if it is among this % of all reviews closest to it (matched_aggregates)"
    )


class CompiledPlan(BaseModel):
    """The final compiled execution plan and metadata."""
    plan_type: str = Field(description="'Plan A' (exact scan) or 'Plan B' (HNSW + BM25 RRF)")
    estimated_candidates: int = Field(description="Number of matching orders (N_cand)")
    probe_latency_ms: float = Field(description="Cardinality probe duration in ms")
    sql: str = Field(description="Executable DuckDB SQL query string")
    parameters: List[Any] = Field(default_factory=list, description="Parameterized query arguments")


class PhysicalPlanner:
    """
    Cost-based optimizer switching at N_cand = settings.PLAN_A_CARDINALITY_LIMIT.
    Plan A is exact (100% recall by construction). Plan B sizes its ANN/BM25 candidate pools
    by filter selectivity so that enough candidates survive the relational filter.
    """

    # Plan B wants ~POOL_TARGET filtered survivors per retriever; pool = POOL_TARGET / selectivity.
    POOL_TARGET = 100
    MIN_POOL = 150
    MAX_POOL = 2000

    def __init__(self, probe: Optional[CardinalityProbe] = None, threshold: int = settings.PLAN_A_CARDINALITY_LIMIT):
        self.probe = probe or CardinalityProbe()
        self.threshold = threshold

    @staticmethod
    def _clean_bm25_keywords(text: str) -> str:
        """Extract clean alphanumeric keywords suitable for DuckDB FTS BM25."""
        words = re.findall(r'\b[a-zA-Z0-9_À-ÿ]{3,}\b', text)
        return " ".join(words[:6]) if words else text

    @classmethod
    def pool_size(cls, card_est: CardinalityEstimate) -> int:
        selectivity = max(card_est.estimated_candidates, 1) / max(card_est.total_orders, 1)
        return min(cls.MAX_POOL, max(cls.MIN_POOL, math.ceil(cls.POOL_TARGET / selectivity)))

    def plan(
        self,
        request: HybridQueryRequest,
        query_vector: List[float],
        conn: duckdb.DuckDBPyConnection,
        force_plan: Optional[str] = None
    ) -> CompiledPlan:
        """Compile the cheaper physical plan. request.sql_filter must already be validated."""
        card_est = self.probe.estimate(conn, request.sql_filter, threshold=self.threshold)
        joins = get_required_joins(request.sql_filter)

        if (force_plan or card_est.recommended_plan) == "Plan A":
            return self._compile_plan_a(request, joins, query_vector, card_est)
        bm25_text = request.bm25_keywords or self._clean_bm25_keywords(request.semantic_query)
        return self._compile_plan_b(request, joins, query_vector, bm25_text, card_est)

    def _compile_plan_a(
        self,
        request: HybridQueryRequest,
        joins: str,
        query_vector: List[float],
        card_est: CardinalityEstimate
    ) -> CompiledPlan:
        """Plan A: relational pushdown, then exact cosine distance over the filtered reviews."""
        sql = f"""
        WITH candidate_orders AS (
            SELECT DISTINCT orders.order_id
            FROM orders
            {joins}
            WHERE {request.sql_filter}
        )
        SELECT
            r.review_id,
            r.order_id,
            r.review_score,
            COALESCE(r.review_comment_message, '') AS review_comment_message,
            ROUND(array_cosine_distance(r.embedding, ?::FLOAT[768]), 4) AS distance
        FROM order_reviews r
        JOIN candidate_orders c ON r.order_id = c.order_id
        WHERE r.embedding IS NOT NULL
        ORDER BY array_cosine_distance(r.embedding, ?::FLOAT[768]) ASC, r.review_id
        LIMIT {request.limit};
        """
        return CompiledPlan(
            plan_type="Plan A",
            estimated_candidates=card_est.estimated_candidates,
            probe_latency_ms=card_est.probe_latency_ms,
            sql=sql,
            parameters=[query_vector, query_vector]
        )

    def _compile_plan_b(
        self,
        request: HybridQueryRequest,
        joins: str,
        query_vector: List[float],
        bm25_text: str,
        card_est: CardinalityEstimate
    ) -> CompiledPlan:
        """
        Plan B: HNSW top-k and BM25 top-k (pool sized by selectivity), fused with
        Reciprocal Rank Fusion (k=60), then filtered by the relational predicate.
        The vector CTE is a plain ORDER BY distance LIMIT k so DuckDB uses the HNSW index;
        ranking happens in a separate CTE because a window function there forces a full scan.
        """
        pool = self.pool_size(card_est)
        sql = f"""
        WITH vector_hits AS (
            SELECT review_id, array_cosine_distance(embedding, ?::FLOAT[768]) AS dist
            FROM order_reviews
            ORDER BY dist
            LIMIT {pool}
        ),
        vector_matches AS (
            SELECT review_id, ROW_NUMBER() OVER (ORDER BY dist, review_id) AS vec_rank
            FROM vector_hits
            WHERE dist IS NOT NULL
        ),
        bm25_matches AS (
            SELECT review_id, ROW_NUMBER() OVER (ORDER BY score DESC, review_id) AS bm25_rank
            FROM (
                SELECT review_id, fts_main_order_reviews.match_bm25(review_id, ?) AS score
                FROM order_reviews
            )
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT {pool}
        ),
        rrf_fused AS (
            SELECT
                COALESCE(v.review_id, b.review_id) AS review_id,
                COALESCE(1.0 / (60.0 + v.vec_rank), 0.0) + COALESCE(1.0 / (60.0 + b.bm25_rank), 0.0) AS rrf_score
            FROM vector_matches v
            FULL OUTER JOIN bm25_matches b ON v.review_id = b.review_id
        )
        SELECT DISTINCT
            r.review_id,
            r.order_id,
            r.review_score,
            COALESCE(r.review_comment_message, '') AS review_comment_message,
            ROUND(f.rrf_score, 5) AS score
        FROM rrf_fused f
        JOIN order_reviews r ON r.review_id = f.review_id
        JOIN orders ON r.order_id = orders.order_id
        {joins}
        WHERE {request.sql_filter}
        ORDER BY score DESC, r.review_id
        LIMIT {request.limit};
        """
        return CompiledPlan(
            plan_type="Plan B",
            estimated_candidates=card_est.estimated_candidates,
            probe_latency_ms=card_est.probe_latency_ms,
            sql=sql,
            parameters=[query_vector, bm25_text]
        )
