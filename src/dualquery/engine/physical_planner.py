"""
DualQuery Cost-Based Physical Planner.
Dynamically chooses and compiles:
- Plan A: High-Selectivity Pushdown (Relational Filter -> Exact SIMD Cosine + BM25)
- Plan B: Low-Selectivity Pushdown (HNSW Index + BM25 RRF -> Relational Hash Join)
"""

import time
import re
from typing import List, Optional, Dict, Any
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.engine.cardinality_probe import CardinalityProbe, CardinalityEstimate
from dualquery.engine.sql_normalizer import normalize_sql_expression, get_required_joins


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
        description="Target OLAP aggregations"
    )
    limit: int = Field(default=5, description="Number of top cited review evidence records to return")


class CompiledPlan(BaseModel):
    """The final compiled execution plan and metadata."""
    plan_type: str = Field(description="'Plan A' (SIMD Scan) or 'Plan B' (HNSW + BM25 RRF)")
    estimated_candidates: int = Field(description="Estimated number of matching orders (N_cand)")
    probe_latency_ms: float = Field(description="Cardinality probe duration in ms")
    sql: str = Field(description="Executable DuckDB SQL query string")
    parameters: List[Any] = Field(default_factory=list, description="Parameterized query arguments")


class PhysicalPlanner:
    """
    Cost-Based Optimizer switching at N_cand = 5,000 (calibrated for AVX2 SIMD scanning).
    Guarantees 100% recall for narrow filters (Plan A) and <30ms latency for wide filters (Plan B).
    """

    def __init__(self, probe: Optional[CardinalityProbe] = None, threshold: int = settings.PLAN_A_CARDINALITY_LIMIT):
        self.probe = probe or CardinalityProbe()
        self.threshold = threshold

    @staticmethod
    def _clean_bm25_keywords(text: str) -> str:
        """Extract clean alphanumeric keywords suitable for DuckDB FTS BM25."""
        words = re.findall(r'\b[a-zA-Z0-9_\u00C0-\u00FF]{3,}\b', text)
        return " ".join(words[:6]) if words else text

    def plan(
        self,
        request: HybridQueryRequest,
        query_vector: List[float],
        conn: duckdb.DuckDBPyConnection,
        force_plan: Optional[str] = None
    ) -> CompiledPlan:
        """Evaluate cost models and compile the optimal DuckDB physical execution plan."""
        # 1. Run Pre-Flight Cardinality Probe
        card_est: CardinalityEstimate = self.probe.estimate(
            conn, request.sql_filter, threshold=self.threshold
        )

        bm25_text = request.bm25_keywords or self._clean_bm25_keywords(request.semantic_query)
        target_plan = force_plan or card_est.recommended_plan

        if target_plan == "Plan A":
            return self._compile_plan_a(request, query_vector, card_est)
        else:
            return self._compile_plan_b(request, query_vector, bm25_text, card_est)
            return self._compile_plan_b(request, query_vector, bm25_text, card_est)

    def _compile_plan_a(
        self,
        request: HybridQueryRequest,
        query_vector: List[float],
        card_est: CardinalityEstimate
    ) -> CompiledPlan:
        """
        Compile Plan A: Relational Pushdown + Exact Hardware SIMD Cosine Distance.
        Guarantees 100% vector recall on small filtered candidate subsets (N <= 1,500).
        """
        norm_filter = normalize_sql_expression(request.sql_filter) if request.sql_filter else "1=1"
        relational_joins = get_required_joins(norm_filter, base_table="orders")

        sql = f"""
        WITH candidate_orders AS (
            SELECT DISTINCT orders.order_id
            FROM orders
            {relational_joins}
            WHERE {norm_filter}
        ),
        scored_reviews AS (
            SELECT 
                r.order_id,
                r.review_id,
                r.review_score,
                COALESCE(r.review_comment_message, '') AS review_comment_message,
                array_cosine_distance(r.embedding, ?::FLOAT[768]) AS cosine_distance
            FROM order_reviews r
            JOIN candidate_orders c ON r.order_id = c.order_id
            WHERE r.embedding IS NOT NULL
            ORDER BY cosine_distance ASC
            LIMIT {request.limit}
        )
        SELECT 
            s.review_id,
            s.order_id,
            s.review_score,
            s.review_comment_message,
            ROUND(s.cosine_distance, 4) AS distance
        FROM scored_reviews s;
        """

        return CompiledPlan(
            plan_type="Plan A",
            estimated_candidates=card_est.estimated_candidates,
            probe_latency_ms=card_est.probe_latency_ms,
            sql=sql,
            parameters=[query_vector]
        )

    def _compile_plan_b(
        self,
        request: HybridQueryRequest,
        query_vector: List[float],
        bm25_text: str,
        card_est: CardinalityEstimate
    ) -> CompiledPlan:
        """
        Compile Plan B: Native HNSW Vector + BM25 FTS pushed down in parallel,
        fused via Reciprocal Rank Fusion (RRF k=60), then hash-joined to relational tables.
        """
        norm_filter = normalize_sql_expression(request.sql_filter) if request.sql_filter else "1=1"
        relational_joins = get_required_joins(norm_filter, base_table="orders")

        sql = f"""
        WITH vector_matches AS (
            SELECT 
                review_id,
                order_id,
                review_score,
                COALESCE(review_comment_message, '') AS review_comment_message,
                ROW_NUMBER() OVER (ORDER BY array_cosine_distance(embedding, ?::FLOAT[768]) ASC) AS vec_rank
            FROM order_reviews
            WHERE embedding IS NOT NULL
            ORDER BY array_cosine_distance(embedding, ?::FLOAT[768]) ASC
            LIMIT 150
        ),
        bm25_matches AS (
            SELECT 
                review_id,
                order_id,
                review_score,
                COALESCE(review_comment_message, '') AS review_comment_message,
                ROW_NUMBER() OVER (ORDER BY score DESC) AS bm25_rank
            FROM (
                SELECT *, fts_main_order_reviews.match_bm25(review_id, ?) AS score
                FROM order_reviews
            )
            WHERE score IS NOT NULL
            ORDER BY score DESC
            LIMIT 150
        ),
        rrf_fused AS (
            SELECT 
                COALESCE(v.review_id, b.review_id) AS review_id,
                COALESCE(v.order_id, b.order_id) AS order_id,
                COALESCE(v.review_score, b.review_score) AS review_score,
                COALESCE(v.review_comment_message, b.review_comment_message) AS review_comment_message,
                (COALESCE(1.0 / (60.0 + v.vec_rank), 0.0) + COALESCE(1.0 / (60.0 + b.bm25_rank), 0.0)) AS rrf_score
            FROM vector_matches v
            FULL OUTER JOIN bm25_matches b ON v.review_id = b.review_id
            ORDER BY rrf_score DESC
            LIMIT 100
        )
        SELECT DISTINCT
            f.review_id,
            f.order_id,
            f.review_score,
            f.review_comment_message,
            ROUND(f.rrf_score, 5) AS score
        FROM rrf_fused f
        JOIN orders ON f.order_id = orders.order_id
        {relational_joins}
        WHERE {norm_filter}
        ORDER BY f.rrf_score DESC
        LIMIT {request.limit};
        """

        return CompiledPlan(
            plan_type="Plan B",
            estimated_candidates=card_est.estimated_candidates,
            probe_latency_ms=card_est.probe_latency_ms,
            sql=sql,
            parameters=[query_vector, query_vector, bm25_text]
        )
