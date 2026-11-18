"""
DualQuery Engine Executor.
Orchestrates:
1. AST validation of the SQL filter (RBAC, PII shield, row-level security)
2. Embedding Generation
3. Cardinality-Aware Physical Planning (Plan A vs Plan B)
4. DuckDB In-Process Execution, fan-out-safe OLAP metrics, and PII scrubbing
"""

import math
import time
import logging
import threading
from pathlib import Path
from typing import Optional, List, Dict, Any
import duckdb
import sqlglot

from dualquery.config import settings
from dualquery.ingestion.embeddings import get_embedding_provider, EmbeddingProvider
from dualquery.engine.physical_planner import PhysicalPlanner, HybridQueryRequest, CompiledPlan
from dualquery.engine.sql_normalizer import normalize_sql_expression, get_required_joins
from dualquery.security.ast_guard import ASTGuard, SecurityPolicy, ANALYST_POLICY
from dualquery.security.pii_vault import PIIVault

from dualquery.engine.models import QueryCitation, QueryExecutionResult

logger = logging.getLogger(__name__)

# Tables that hold several rows per order. Joining two of them multiplies rows and inflates SUM/COUNT.
ONE_TO_MANY_TABLES = ("order_payments", "order_items", "order_reviews")


class HybridExecutor:
    """End-to-end execution runtime for DualQuery."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedder: Optional[EmbeddingProvider] = None,
        planner: Optional[PhysicalPlanner] = None,
        guard: Optional[ASTGuard] = None,
        vault: Optional[PIIVault] = None,
    ):
        self.db_path = str(db_path or settings.DUCKDB_PATH)
        self.embedder = embedder or get_embedding_provider("auto")
        self.planner = planner or PhysicalPlanner()
        self.guard = guard or ASTGuard()
        self.vault = vault or PIIVault()
        self._db: Optional[duckdb.DuckDBPyConnection] = None
        self._db_lock = threading.Lock()

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Return a new cursor on a shared read-only database handle, opened once (saves ~120 ms per turn).
        Cursors are safe to use from different threads. The handle is sandboxed from the host.
        """
        with self._db_lock:
            if self._db is None:
                if not Path(self.db_path).exists():
                    raise FileNotFoundError(
                        f"DuckDB database not found at {self.db_path}. Build it with `dualquery ingest-olist` "
                        "or set DUALQUERY_DB_PATH."
                    )
                db = duckdb.connect(self.db_path, read_only=True)
                db.execute("LOAD vss; LOAD fts;")
                # Defense in depth behind the AST guard: no file/network access, and no way to turn it back on.
                db.execute("SET enable_external_access = false; SET lock_configuration = true;")
                self._db = db
        return self._db.cursor()

    def validate_filter(self, sql_filter: str, policy: SecurityPolicy) -> str:
        """
        Run the filter through the AST guard and return the WHERE clause that must be executed:
        aliases expanded and the role's row-level security predicate appended.
        """
        norm = normalize_sql_expression(sql_filter or "").strip() or "1=1"
        probe_sql = f"SELECT * FROM orders {get_required_joins(norm)} WHERE {norm}"
        validated = sqlglot.parse_one(self.guard.validate_and_rewrite(probe_sql, policy=policy), read="duckdb")
        return validated.args["where"].this.sql(dialect="duckdb")

    def validate_metric(self, metric: str, policy: SecurityPolicy) -> str:
        """Validate one aggregation expression and reject ones that would double count."""
        norm = normalize_sql_expression(metric).strip()
        joins = get_required_joins(norm)
        fanout = [t for t in ONE_TO_MANY_TABLES if f"JOIN {t} " in joins]
        if len(fanout) > 1:
            raise ValueError(
                f"Metric '{metric}' combines {fanout}; joining two one-to-many tables double counts. "
                "Split it into one metric per table."
            )
        self.guard.validate_and_rewrite(f"SELECT {norm} FROM orders {joins}", policy=policy)
        return norm

    def execute(
        self,
        request: HybridQueryRequest,
        policy: Optional[SecurityPolicy] = None,
        force_plan: Optional[str] = None,
        tokenize_citations: bool = True,
    ) -> QueryExecutionResult:
        """
        Execute a compound hybrid query and extract audited telemetry.
        The validated filter (including row-level security) is the only filter that reaches DuckDB.
        """
        total_start = time.perf_counter()
        active_policy = policy or ANALYST_POLICY

        safe_filter = self.validate_filter(request.sql_filter, active_policy)
        metrics = [self.validate_metric(m, active_policy) for m in request.aggregations]
        safe_request = request.model_copy(update={"sql_filter": safe_filter})

        conn = self.connect()
        try:
            query_vector = self.embedder.embed_query(request.semantic_query)
            compiled: CompiledPlan = self.planner.plan(safe_request, query_vector, conn, force_plan=force_plan)

            engine_start = time.perf_counter()
            raw_rows = conn.execute(compiled.sql, compiled.parameters).fetchall()
            engine_latency_ms = (time.perf_counter() - engine_start) * 1000

            citations = [
                self.vault.scrub_citation(
                    QueryCitation(review_id=r[0], order_id=r[1], review_score=r[2], review_comment=r[3], score_or_distance=r[4]),
                    tokenize_ids=tokenize_citations,
                )
                for r in raw_rows
            ]

            matched_ids = self._matched_order_ids(conn, safe_filter, query_vector, request.match_top_percent)
            return QueryExecutionResult(
                plan_type=compiled.plan_type,
                estimated_candidates=compiled.estimated_candidates,
                probe_latency_ms=compiled.probe_latency_ms,
                engine_latency_ms=round(engine_latency_ms, 2),
                total_latency_ms=round((time.perf_counter() - total_start) * 1000, 2),
                aggregates=self._compute_aggregates(conn, safe_filter, metrics, active_policy),
                matched_orders=len(matched_ids),
                matched_aggregates=self._compute_aggregates(conn, safe_filter, metrics, active_policy, matched_ids),
                citations=citations,
            )
        finally:
            conn.close()

    @staticmethod
    def _matched_order_ids(
        conn: duckdb.DuckDBPyConnection, safe_filter: str, query_vector: List[float], top_percent: float
    ) -> List[str]:
        """
        Filtered orders with at least one review among the top_percent % of all reviews closest to the query.
        A per-query rank cutoff instead of a fixed distance, because absolute cosine distances shift a lot
        between queries (measured nearest-review distance ranges 0.26-0.49 on the golden set).
        The global top-k comes from the HNSW index, so the result is approximate.
        """
        embedded = conn.execute("SELECT COUNT(embedding) FROM order_reviews").fetchone()[0]
        k = max(1, math.ceil(embedded * top_percent / 100.0))
        rows = conn.execute(
            f"""
            WITH hits AS (
                SELECT order_id, array_cosine_distance(embedding, ?::FLOAT[768]) AS d
                FROM order_reviews
                ORDER BY d
                LIMIT {k}
            )
            SELECT DISTINCT hits.order_id
            FROM hits
            WHERE hits.d IS NOT NULL
              AND hits.order_id IN (
                  SELECT orders.order_id FROM orders {get_required_joins(safe_filter)} WHERE {safe_filter}
              )
            """,
            [query_vector],
        ).fetchall()
        return [r[0] for r in rows]

    @staticmethod
    def _compute_aggregates(
        conn: duckdb.DuckDBPyConnection,
        safe_filter: str,
        metrics: List[str],
        policy: SecurityPolicy,
        only_order_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compute each metric over the distinct set of matching orders, joining only the tables the
        metric itself needs, so one-to-many joins in the filter can never multiply rows.
        """
        results: Dict[str, Any] = {}
        for metric in metrics:
            metric_joins = get_required_joins(metric)
            where = [f"orders.order_id IN (SELECT orders.order_id FROM orders {get_required_joins(safe_filter)} WHERE {safe_filter})"]
            params: List[Any] = []
            if only_order_ids is not None:
                where.append("orders.order_id IN (SELECT unnest(?::VARCHAR[]))")
                params.append(only_order_ids)
            # Row-level security also scopes the rows of its own table (e.g. only this seller's items).
            if policy.rls_predicate and get_required_joins(policy.rls_predicate) in metric_joins:
                where.append(f"({policy.rls_predicate})")
            cur = conn.execute(f"SELECT {metric} FROM orders {metric_joins} WHERE {' AND '.join(where)}", params)
            results.update(zip([d[0] for d in cur.description], cur.fetchone()))
        return results
