"""
DualQuery Pre-Flight Cardinality Probe.
Executes an ultra-fast (<2ms) COUNT(*) probe on candidate relational predicates
to estimate candidate cardinality N_cand before query compilation.
"""

import time
import re
import logging
from typing import Optional, Tuple
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.engine.sql_normalizer import normalize_sql_expression, get_required_joins

logger = logging.getLogger(__name__)


def normalize_sql_filter(filter_str: str) -> str:
    """Backward-compatible wrapper for normalize_sql_expression."""
    return normalize_sql_expression(filter_str)


class CardinalityEstimate(BaseModel):
    """Result of a pre-flight cardinality probe."""
    estimated_candidates: int = Field(description="Estimated number of matching orders (N_cand)")
    probe_latency_ms: float = Field(description="Execution latency of the probe query in milliseconds")
    recommended_plan: str = Field(description="'Plan A' (SIMD Scan) or 'Plan B' (HNSW Pushdown)")
    threshold_limit: int = Field(description="Plan A cardinality limit threshold")


class CardinalityProbe:
    """
    Sub-2ms Relational Selectivity Estimator.
    Constructs a lightweight COUNT query joining only the tables referenced in the relational filter.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or settings.DUCKDB_PATH)

    def estimate(
        self,
        conn: duckdb.DuckDBPyConnection,
        sql_filter: str,
        threshold: int = settings.PLAN_A_CARDINALITY_LIMIT
    ) -> CardinalityEstimate:
        """
        Probe DuckDB to determine how many distinct orders satisfy the SQL filter.
        """
        cleaned_filter = normalize_sql_expression(sql_filter.strip()) if sql_filter else "1=1"
        if not cleaned_filter:
            cleaned_filter = "1=1"

        join_clause = get_required_joins(cleaned_filter, base_table="orders")

        probe_query = f"""
            SELECT COUNT(DISTINCT orders.order_id)
            FROM orders
            {join_clause}
            WHERE {cleaned_filter};
        """

        start = time.perf_counter()
        try:
            count = conn.execute(probe_query).fetchone()[0]
        except Exception as e:
            logger.error(f"Cardinality probe query failed: {probe_query}. Error: {e}")
            # Safe fallback: assume large cardinality to default to Plan B
            count = threshold + 1

        latency_ms = (time.perf_counter() - start) * 1000

        recommended_plan = "Plan A" if count <= threshold else "Plan B"

        return CardinalityEstimate(
            estimated_candidates=count,
            probe_latency_ms=round(latency_ms, 2),
            recommended_plan=recommended_plan,
            threshold_limit=threshold
        )
