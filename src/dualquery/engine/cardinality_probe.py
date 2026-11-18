"""
DualQuery Pre-Flight Cardinality Probe.
Runs a COUNT(DISTINCT order_id) over the validated relational filter
to estimate candidate cardinality N_cand before query compilation.
"""

import time
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.engine.sql_normalizer import get_required_joins


class CardinalityEstimate(BaseModel):
    """Result of a pre-flight cardinality probe."""
    estimated_candidates: int = Field(description="Number of distinct orders matching the filter (N_cand)")
    total_orders: int = Field(description="Total orders in the database")
    probe_latency_ms: float = Field(description="Execution latency of the probe query in milliseconds")
    recommended_plan: str = Field(description="'Plan A' (exact scan) or 'Plan B' (HNSW + BM25 RRF)")
    threshold_limit: int = Field(description="Plan A cardinality limit threshold")


class CardinalityProbe:
    """Relational selectivity estimator joining only the tables the filter references."""

    def estimate(
        self,
        conn: duckdb.DuckDBPyConnection,
        sql_filter: str,
        threshold: int = settings.PLAN_A_CARDINALITY_LIMIT
    ) -> CardinalityEstimate:
        """Count distinct orders that satisfy an already-validated SQL filter. Errors propagate."""
        start = time.perf_counter()
        count, total = conn.execute(f"""
            SELECT
                (SELECT COUNT(DISTINCT orders.order_id) FROM orders {get_required_joins(sql_filter)} WHERE {sql_filter}),
                (SELECT COUNT(*) FROM orders)
        """).fetchone()
        latency_ms = (time.perf_counter() - start) * 1000

        return CardinalityEstimate(
            estimated_candidates=count,
            total_orders=total,
            probe_latency_ms=round(latency_ms, 2),
            recommended_plan="Plan A" if count <= threshold else "Plan B",
            threshold_limit=threshold
        )
