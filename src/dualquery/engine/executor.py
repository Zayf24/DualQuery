"""
DualQuery Engine Executor.
Orchestrates:
1. Literal Grounding
2. Embedding Generation
3. Cardinality-Aware Physical Planning (Plan A vs Plan B)
4. DuckDB In-Process Execution & Telemetry Extraction
"""

import time
import logging
from typing import Optional, List, Dict, Any
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.ingestion.embeddings import get_embedding_provider, EmbeddingProvider
from dualquery.engine.literal_grounder import LiteralGrounder
from dualquery.engine.cardinality_probe import CardinalityProbe
from dualquery.engine.physical_planner import PhysicalPlanner, HybridQueryRequest, CompiledPlan
from dualquery.engine.sql_normalizer import normalize_sql_expression, get_required_joins
from dualquery.security.ast_guard import ASTGuard, SecurityPolicy, ANALYST_POLICY
from dualquery.security.pii_vault import PIIVault

from dualquery.engine.models import QueryCitation, QueryExecutionResult

logger = logging.getLogger(__name__)


class HybridExecutor:
    """End-to-end execution runtime for DualQuery."""

    def __init__(
        self,
        db_path: Optional[str] = None,
        embedder: Optional[EmbeddingProvider] = None,
        planner: Optional[PhysicalPlanner] = None,
        grounder: Optional[LiteralGrounder] = None,
        guard: Optional[ASTGuard] = None,
        vault: Optional[PIIVault] = None,
    ):
        self.db_path = str(db_path or settings.DUCKDB_PATH)
        self.embedder = embedder or get_embedding_provider("auto")
        self.planner = planner or PhysicalPlanner()
        self.grounder = grounder or LiteralGrounder()
        self.guard = guard or ASTGuard()
        self.vault = vault or PIIVault()

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Create read-only connection to DuckDB with extensions."""
        conn = duckdb.connect(self.db_path, read_only=True)
        conn.execute("LOAD vss; LOAD fts;")
        return conn

    def execute(
        self,
        request: HybridQueryRequest,
        policy: Optional[SecurityPolicy] = None,
        force_plan: Optional[str] = None,
        tokenize_citations: bool = True,
    ) -> QueryExecutionResult:
        """
        Execute a compound hybrid query and extract audited telemetry.
        Enforces 5-gate AST security validation and in-process PII scrubbing.
        """
        total_start = time.perf_counter()

        # 0. Enforce Zero-Trust Security Validation via AST Guard
        active_policy = policy or ANALYST_POLICY
        if request.sql_filter and request.sql_filter.strip() != "1=1":
            # Normalize short aliases and build required join tree before validation
            norm_filter = normalize_sql_expression(request.sql_filter)
            filter_joins = get_required_joins(norm_filter, base_table="orders")
            probe_sql = f"SELECT * FROM orders {filter_joins} WHERE {norm_filter}"
            self.guard.validate_and_rewrite(probe_sql, policy=active_policy)

        if request.aggregations:
            norm_agg = normalize_sql_expression(", ".join(request.aggregations))
            agg_joins = get_required_joins(norm_agg, base_table="orders")
            probe_agg = f"SELECT {norm_agg} FROM orders {agg_joins}"
            self.guard.validate_and_rewrite(probe_agg, policy=active_policy)

        conn = self.connect()

        try:
            # 1. Generate Query Vector Embedding
            query_vector = self.embedder.embed_query(request.semantic_query)

            # 2. Compile Physical Plan via Optimizer
            compiled: CompiledPlan = self.planner.plan(
                request, query_vector, conn, force_plan=force_plan
            )

            # 3. Execute Compiled Plan in DuckDB
            engine_start = time.perf_counter()
            cursor = conn.execute(compiled.sql, compiled.parameters)
            raw_rows = cursor.fetchall()
            engine_latency_ms = (time.perf_counter() - engine_start) * 1000

            raw_citations = [
                QueryCitation(
                    review_id=r[0],
                    order_id=r[1],
                    review_score=r[2],
                    review_comment=r[3],
                    score_or_distance=r[4]
                )
                for r in raw_rows
            ]

            # 4. In-Process PII Scrubbing and HMAC Tokenization
            citations = [
                self.vault.scrub_citation(c, tokenize_ids=tokenize_citations)
                for c in raw_citations
            ]

            # 5. Compute OLAP Aggregations over Candidate Orders
            agg_results = self._compute_aggregates(conn, request)

            total_latency_ms = (time.perf_counter() - total_start) * 1000

            return QueryExecutionResult(
                plan_type=compiled.plan_type,
                estimated_candidates=compiled.estimated_candidates,
                probe_latency_ms=compiled.probe_latency_ms,
                engine_latency_ms=round(engine_latency_ms, 2),
                total_latency_ms=round(total_latency_ms, 2),
                aggregates=agg_results,
                citations=citations
            )

        finally:
            conn.close()

    def _compute_aggregates(
        self,
        conn: duckdb.DuckDBPyConnection,
        request: HybridQueryRequest
    ) -> Dict[str, Any]:
        """Execute mathematical aggregations across matching candidate orders."""
        if not request.aggregations:
            return {}

        agg_clause = ", ".join(request.aggregations)
        norm_filter = normalize_sql_expression(request.sql_filter) if request.sql_filter else "1=1"
        norm_agg = normalize_sql_expression(agg_clause)
        join_str = get_required_joins([norm_filter, norm_agg], base_table="orders")

        agg_sql = f"""
            SELECT {norm_agg}
            FROM orders
            {join_str}
            WHERE {norm_filter};
        """
        try:
            res = conn.execute(agg_sql).fetchone()
            columns = [desc[0] for desc in conn.description]
            return {columns[i]: res[i] for i in range(len(columns))}
        except Exception as e:
            logger.warning(f"Aggregate computation failed: {e}")
            return {}
