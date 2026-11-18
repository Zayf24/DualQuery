"""
End-to-end engine tests on the tiny fixture database (see conftest.py for the data).
These cover the paths the audit found broken: row-level security, fan-out-safe metrics,
semantic-scoped metrics, Plan B index use and recall, error propagation, and the DB sandbox.
"""

from decimal import Decimal

import duckdb
import pytest

from dualquery.engine.cardinality_probe import CardinalityEstimate
from dualquery.engine.executor import HybridExecutor
from dualquery.engine.physical_planner import HybridQueryRequest, PhysicalPlanner
from dualquery.security.ast_guard import (
    ASTSecurityViolation,
    SecurityPolicy,
    SUPPORT_AGENT_POLICY,
)

SELLER_S1 = SecurityPolicy(
    role_name="seller_s1",
    allowed_tables={"orders", "order_items", "products", "order_reviews"},
    rls_predicate="order_items.seller_id = 'S1'",
)


def req(sql_filter="1=1", semantic="broken", metrics=None, **kw):
    return HybridQueryRequest(
        sql_filter=sql_filter, semantic_query=semantic,
        aggregations=metrics if metrics is not None else ["COUNT(*) AS n"], **kw
    )


def raw_ids(executor, result_citations):
    """Map tokenized citation order ids back to real ids via the same vault."""
    lookup = {executor.vault.tokenize_identifier(o, "ORD"): o for o in ("O1", "O2", "O3", "O4")}
    return {lookup[c.order_id] for c in result_citations}


# --- Finding 2: metrics must not be multiplied by one-to-many joins -------------------------

def test_sum_is_not_inflated_by_item_join(executor):
    # Filter joins order_items (O1 has 2 items) while the metric sums payments (O1 has 2 payments).
    res = executor.execute(req("order_items.price > 0", metrics=["SUM(order_payments.payment_value) AS paid"]))
    assert res.aggregates["paid"] == Decimal("185.00")  # 30 + 100 + 50 + 5; the old join gave 215 (O1 counted twice)


def test_count_star_counts_orders_not_join_rows(executor):
    res = executor.execute(req("1=1", metrics=["COUNT(*) AS n", "SUM(order_payments.payment_value) AS paid"]))
    assert res.aggregates == {"n": 4, "paid": Decimal("185.00")}


def test_metric_on_items_counts_each_item_once(executor):
    res = executor.execute(req("customers.customer_state = 'SP'", metrics=["SUM(order_items.price) AS items"]))
    assert res.aggregates["items"] == Decimal("35.00")  # O1: 10 + 20, O4: 5


def test_metric_mixing_two_one_to_many_tables_is_rejected(executor):
    with pytest.raises(ValueError, match="double counts"):
        executor.execute(req(metrics=["SUM(order_payments.payment_value) + SUM(order_items.price)"]))


def test_filter_matching_nothing_returns_zero_and_null(executor):
    res = executor.execute(req("orders.order_status = 'shipped'", metrics=["COUNT(*) AS n", "SUM(order_items.price) AS s"]))
    assert res.aggregates == {"n": 0, "s": None}
    assert res.citations == [] and res.matched_orders == 0


# --- Finding 1: row-level security reaches every executed query --------------------------

def test_rls_restricts_citations_and_metrics(executor):
    res = executor.execute(
        req("1=1", metrics=["COUNT(*) AS n", "COUNT(DISTINCT order_items.seller_id) AS sellers", "SUM(order_items.price) AS items"]),
        policy=SELLER_S1,
    )
    assert res.estimated_candidates == 2  # O1, O2 contain an S1 item
    assert res.aggregates == {"n": 2, "sellers": 1, "items": Decimal("110.00")}  # S2's 20.00 in O1 excluded
    assert raw_ids(executor, res.citations) <= {"O1", "O2"}


def test_rls_survives_or_in_user_filter(executor):
    res = executor.execute(
        req("orders.order_status = 'canceled' OR orders.order_status = 'delivered'"), policy=SELLER_S1
    )
    assert res.estimated_candidates == 2 and res.aggregates["n"] == 2


@pytest.mark.parametrize("plan", ["Plan A", "Plan B"])
def test_rls_applies_to_both_plans(executor, plan):
    res = executor.execute(req("1=1", limit=10), policy=SELLER_S1, force_plan=plan)
    assert res.plan_type == plan
    assert raw_ids(executor, res.citations) and raw_ids(executor, res.citations) <= {"O1", "O2"}


def test_validated_filter_contains_rls_and_expanded_aliases(executor):
    safe = executor.validate_filter("c.customer_state = 'SP'", SELLER_S1.model_copy(update={"allowed_tables": SELLER_S1.allowed_tables | {"customers"}}))
    assert "customers.customer_state = 'SP'" in safe and "order_items.seller_id = 'S1'" in safe


# --- Finding 3: matched scope only counts orders whose reviews match the query ------------

def test_matched_scope_counts_only_semantic_matches(executor):
    # 3 of 5 reviews say "broken" (R1, R5 on O1; R4 on O4). Top 50% of reviews by similarity = those 3.
    res = executor.execute(req("1=1", "broken", ["COUNT(*) AS n", "SUM(order_payments.payment_value) AS paid"], match_top_percent=50))
    assert res.aggregates == {"n": 4, "paid": Decimal("185.00")}
    assert res.matched_orders == 2
    assert res.matched_aggregates == {"n": 2, "paid": Decimal("35.00")}


def test_matched_scope_respects_filter(executor):
    res = executor.execute(req("customers.customer_state = 'RJ'", "broken", match_top_percent=50))
    assert res.matched_orders == 0 and res.matched_aggregates == {"n": 0}


# --- Findings 6, 7: Plan B uses HNSW and sizes its pool by selectivity --------------------

def test_plan_b_vector_leg_uses_hnsw_index(executor, fake_embedder):
    conn = executor.connect()
    try:
        compiled = executor.planner.plan(req("1=1"), fake_embedder.embed_query("broken"), conn, force_plan="Plan B")
        plan_text = conn.execute("EXPLAIN " + compiled.sql, compiled.parameters).fetchall()[0][1]
    finally:
        conn.close()
    assert "HNSW_INDEX_SCAN" in plan_text.upper().replace(" ", "_")


def test_plan_b_returns_filtered_hits_ranked_by_rrf(executor):
    res = executor.execute(req("customers.customer_state = 'SP'", "broken", limit=10), force_plan="Plan B")
    assert raw_ids(executor, res.citations) == {"O1", "O4"}
    scores = [c.score_or_distance for c in res.citations]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize("n_cand,expected", [(0, 2000), (1, 2000), (5_000, 1989), (20_000, 498), (99_441, 150)])
def test_pool_size_bounds(n_cand, expected):
    est = CardinalityEstimate(estimated_candidates=n_cand, total_orders=99_441, probe_latency_ms=0,
                              recommended_plan="Plan B", threshold_limit=5000)
    assert PhysicalPlanner.pool_size(est) == expected


def test_plan_a_is_exact_and_ordered(executor):
    res = executor.execute(req("1=1", "broken", limit=3), force_plan="Plan A")
    assert res.plan_type == "Plan A"
    assert raw_ids(executor, res.citations) == {"O1", "O4"}
    assert [c.score_or_distance for c in res.citations] == [0.0, 0.0, 0.0]


def test_citations_are_scrubbed_and_tokenized(executor):
    res = executor.execute(req("orders.order_id = 'O1'", "broken", limit=5))
    text = " ".join(c.review_comment for c in res.citations)
    assert "joao@mail.com" not in text and "[EMAIL_REDACTED]" in text
    assert all(c.order_id.startswith("[ORD_") and c.review_id.startswith("[REV_") for c in res.citations)


# --- Finding 10: failures surface instead of returning empty "success" ---------------------

def test_unknown_metric_column_raises(executor):
    with pytest.raises(duckdb.BinderException):
        executor.execute(req(metrics=["SUM(order_payments.payment_valu)"]))


def test_unknown_filter_column_raises(executor):
    with pytest.raises(duckdb.BinderException):
        executor.execute(req("orders.no_such_column = 1"))


# --- Security through the real executor path ---------------------------------------------

@pytest.mark.parametrize("bad_filter,gate", [
    ("1=1; DROP TABLE orders", 1),
    ("orders.order_id IN (SELECT * FROM read_csv('secrets.csv'))", 2),
    ("length(getenv('PATH')) > 0", 2),
    ("orders.order_id IN (SELECT order_id FROM order_payments)", 3),
    ("customers.customer_unique_id = 'U1'", 4),
])
def test_executor_blocks_malicious_filters(executor, bad_filter, gate):
    with pytest.raises(ASTSecurityViolation) as err:
        executor.execute(req(bad_filter), policy=SUPPORT_AGENT_POLICY)
    assert err.value.gate == gate


def test_executor_blocks_pii_in_metrics(executor):
    with pytest.raises(ASTSecurityViolation) as err:
        executor.execute(req(metrics=["COUNT(DISTINCT customers.customer_unique_id)"]))
    assert err.value.gate == 4


def test_connection_is_sandboxed(executor, tmp_path):
    secret = tmp_path / "secret.csv"
    secret.write_text("a\n1\n")
    conn = executor.connect()
    try:
        with pytest.raises(duckdb.Error):
            conn.execute(f"SELECT * FROM read_csv('{secret.as_posix()}')").fetchall()
        with pytest.raises(duckdb.Error):
            conn.execute("SET enable_external_access = true")
        with pytest.raises(duckdb.Error):
            conn.execute("CREATE TABLE x (a INT)")  # read-only
    finally:
        conn.close()


def test_missing_database_gives_actionable_error(tmp_path, fake_embedder):
    ex = HybridExecutor(db_path=str(tmp_path / "nope.duckdb"), embedder=fake_embedder)
    with pytest.raises(FileNotFoundError, match="ingest-olist"):
        ex.execute(req())
