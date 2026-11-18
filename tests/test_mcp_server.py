"""
MCP server tests: tool discovery, server-fixed role (RBAC), literal grounding, labeled
filter/matched metrics, and error shaping. Runs against the tiny fixture database.
"""

import pytest

import dualquery.mcp.server as server
from dualquery.engine.executor import HybridExecutor
from dualquery.security.ast_guard import ANALYST_POLICY, SUPPORT_AGENT_POLICY


@pytest.fixture
def mcp(tiny_db, fake_embedder, monkeypatch):
    monkeypatch.setattr(server, "_executor", HybridExecutor(db_path=tiny_db, embedder=fake_embedder))
    monkeypatch.setattr(server, "_grounder", None)
    monkeypatch.setattr(server, "ACTIVE_POLICY", ANALYST_POLICY)
    return server.mcp_server


async def call(mcp, tool, args):
    res = await mcp.call_tool(tool, args)
    return res.structured_content["result"]


@pytest.mark.asyncio
async def test_tools_are_registered_without_a_role_argument(mcp):
    tools = {t.name: t for t in await mcp.list_tools()}
    assert set(tools) == {"get_governed_catalog", "resolve_entity_literals", "execute_hybrid_query"}
    for tool in tools.values():
        assert "role" not in tool.input_schema.get("properties", {})


@pytest.mark.asyncio
async def test_catalog_follows_server_role(mcp, monkeypatch):
    assert "order_payments" in (await call(mcp, "get_governed_catalog", {}))["authorized_tables"]
    monkeypatch.setattr(server, "ACTIVE_POLICY", SUPPORT_AGENT_POLICY)
    catalog = await call(mcp, "get_governed_catalog", {})
    assert catalog["role"] == "support_agent" and "order_payments" not in catalog["authorized_tables"]


def test_unknown_role_is_rejected_at_startup():
    with pytest.raises(ValueError, match="Unknown role"):
        server.run_server(role="admin")


@pytest.mark.asyncio
async def test_literal_grounding(mcp):
    res = await call(mcp, "resolve_entity_literals", {"terms": ["Rio de Janeiro", "credit card", "zzzz-unknown"]})
    assert res["entities"]["Rio de Janeiro"]["grounded_value"] == "RJ"
    assert res["entities"]["credit card"]["grounded_value"] == "credit_card"
    assert res["entities"]["zzzz-unknown"]["grounded_value"] is None
    assert res["grounded_count"] == 2


@pytest.mark.asyncio
async def test_execute_returns_labeled_scopes_and_tokenized_citations(mcp):
    res = await call(mcp, "execute_hybrid_query", {
        "sql_filter": "1=1", "semantic_query": "broken", "match_top_percent": 50,
        "aggregations": ["COUNT(*) AS n", "SUM(order_payments.payment_value) AS paid"],
    })
    assert res["status"] == "success"
    assert res["aggregates"]["filter_scope"] == {
        "description": "Every order matching sql_filter", "orders": 4, "metrics": {"n": "4", "paid": "185.00"}}
    assert res["aggregates"]["matched_scope"]["orders"] == 2
    assert res["aggregates"]["matched_scope"]["metrics"] == {"n": "2", "paid": "35.00"}
    assert res["citations"] and all(c["order_id"].startswith("[ORD_") for c in res["citations"])
    assert res["security_governance"]["active_role"] == "analyst"


@pytest.mark.asyncio
async def test_limit_is_capped_at_10(mcp):
    res = await call(mcp, "execute_hybrid_query", {"semantic_query": "broken", "limit": 500})
    assert res["status"] == "success" and res["citations_returned"] <= 10


@pytest.mark.asyncio
async def test_empty_semantic_query_is_rejected(mcp):
    res = await call(mcp, "execute_hybrid_query", {"semantic_query": "  "})
    assert res["status"] == "error" and res["error_type"] == "INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_chained_drop_is_blocked_at_gate_1(mcp):
    res = await call(mcp, "execute_hybrid_query", {"sql_filter": "1=1; DROP TABLE customers;", "semantic_query": "broken"})
    assert res["status"] == "security_violation" and res["gate"] == 1


@pytest.mark.asyncio
async def test_support_agent_cannot_reach_payments(mcp, monkeypatch):
    monkeypatch.setattr(server, "ACTIVE_POLICY", SUPPORT_AGENT_POLICY)
    res = await call(mcp, "execute_hybrid_query", {"sql_filter": "order_payments.payment_value > 500", "semantic_query": "fee"})
    assert res["status"] == "security_violation" and res["gate"] == 3


@pytest.mark.asyncio
async def test_double_counting_metric_returns_actionable_error(mcp):
    res = await call(mcp, "execute_hybrid_query", {
        "semantic_query": "broken", "aggregations": ["SUM(order_payments.payment_value) + SUM(order_items.price)"]})
    assert res["status"] == "execution_error" and "double counts" in res["message"]


@pytest.mark.asyncio
async def test_database_errors_do_not_leak_sql(mcp):
    res = await call(mcp, "execute_hybrid_query", {"semantic_query": "broken", "aggregations": ["SUM(orders.nope)"]})
    assert res["status"] == "execution_error" and res["error_type"] == "BinderException"
    assert "\n" not in res["message"] and len(res["message"]) <= 200
