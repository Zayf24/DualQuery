"""
DualQuery Zero-Trust Model Context Protocol (MCP) Server.
Exposes 3 strongly-typed JSON-RPC 2.0 tools over stdio and SSE/HTTP:
1. get_governed_catalog: Role-scoped catalog inspection.
2. resolve_entity_literals: Pre-SQL fuzzy & semantic literal grounding.
3. execute_hybrid_query: Zero-trust hybrid execution (Plan A/B pushdown, in-process OLAP, local PII redaction).
"""

import sys
import logging
from typing import List, Dict, Any, Optional
from mcp.server.mcpserver import MCPServer

from dualquery.engine.executor import HybridExecutor
from dualquery.engine.physical_planner import HybridQueryRequest
from dualquery.engine.literal_grounder import LiteralGrounder
from dualquery.security.ast_guard import (
    SecurityPolicy,
    ANALYST_POLICY,
    SUPPORT_AGENT_POLICY,
    SELLER_RESTRICTED_POLICY,
    ASTSecurityViolation,
)

logger = logging.getLogger("dualquery.mcp")

# Initialize MCP Server instance
mcp_server = MCPServer(
    name="dualquery-mcp",
    version="0.1.0",
    description="Zero-Trust Hybrid Relational-Semantic Query Engine & MCP Server",
)

# Role policy registry
ROLE_POLICIES: Dict[str, SecurityPolicy] = {
    "analyst": ANALYST_POLICY,
    "support_agent": SUPPORT_AGENT_POLICY,
    "seller_restricted": SELLER_RESTRICTED_POLICY,
}

# Catalog definitions for role-governed schema exposure
BASE_CATALOG = {
    "orders": {
        "description": "Central order records with delivery dates and delay metrics",
        "columns": [
            "order_id (VARCHAR, PK)",
            "customer_id (VARCHAR, FK -> customers.customer_id)",
            "order_status (VARCHAR, e.g. 'delivered', 'shipped', 'canceled')",
            "order_purchase_timestamp (TIMESTAMP)",
            "order_delivered_customer_date (TIMESTAMP)",
            "order_estimated_delivery_date (TIMESTAMP)",
            "delivery_delay_days (INTEGER, positive means late delivery)",
        ],
        "allowed_metrics": ["COUNT(DISTINCT orders.order_id)", "AVG(orders.delivery_delay_days)"],
    },
    "customers": {
        "description": "Customer location attributes (PII unique IDs scrubbed)",
        "columns": [
            "customer_id (VARCHAR, PK)",
            "customer_city (VARCHAR)",
            "customer_state (VARCHAR(2), e.g. 'SP', 'RJ', 'MG')",
        ],
        "allowed_metrics": ["COUNT(DISTINCT customers.customer_id)"],
    },
    "order_payments": {
        "description": "Financial payment transaction records",
        "columns": [
            "order_id (VARCHAR, FK -> orders.order_id)",
            "payment_sequential (INTEGER)",
            "payment_type (VARCHAR, e.g. 'credit_card', 'boleto', 'voucher', 'debit_card')",
            "payment_installments (INTEGER)",
            "payment_value (DECIMAL(10,2))",
        ],
        "allowed_metrics": ["ROUND(SUM(order_payments.payment_value), 2)", "AVG(order_payments.payment_installments)"],
    },
    "order_items": {
        "description": "Product line items, pricing, freight, and seller assignments",
        "columns": [
            "order_id (VARCHAR, FK -> orders.order_id)",
            "order_item_id (INTEGER)",
            "product_id (VARCHAR, FK -> products.product_id)",
            "seller_id (VARCHAR)",
            "price (DECIMAL(10,2))",
            "freight_value (DECIMAL(10,2))",
        ],
        "allowed_metrics": ["SUM(order_items.price)", "AVG(order_items.freight_value)"],
    },
    "products": {
        "description": "Product taxonomy and physical properties",
        "columns": [
            "product_id (VARCHAR, PK)",
            "product_category_name (VARCHAR, Portuguese category enum)",
            "product_category_name_english (VARCHAR, English translation)",
            "product_weight_g (INTEGER)",
        ],
        "allowed_metrics": ["COUNT(DISTINCT products.product_id)"],
    },
    "order_reviews": {
        "description": "Customer feedback reviews, ratings, and 768-d semantic embeddings",
        "columns": [
            "review_id (VARCHAR, PK)",
            "order_id (VARCHAR, FK -> orders.order_id)",
            "review_score (TINYINT, 1 to 5)",
            "review_comment_title (VARCHAR)",
            "review_comment_message (VARCHAR, text feedback)",
            "review_creation_date (TIMESTAMP)",
        ],
        "allowed_metrics": ["AVG(order_reviews.review_score)", "COUNT(order_reviews.review_id)"],
    },
}

# The role is fixed by whoever starts the server (`--role`), never by the model calling the tools.
ACTIVE_POLICY: SecurityPolicy = ANALYST_POLICY

# Shared singletons for high performance
_executor: Optional[HybridExecutor] = None
_grounder: Optional[LiteralGrounder] = None


def get_executor() -> HybridExecutor:
    global _executor
    if _executor is None:
        _executor = HybridExecutor()
    return _executor


def get_grounder() -> LiteralGrounder:
    global _grounder
    if _grounder is None:
        # Reuse the executor's embedder so the semantic tier ('cosmetics' -> 'health_beauty') is active.
        _grounder = LiteralGrounder(embedder=get_executor().embedder)
    return _grounder


# ==============================================================================
# MCP Tool 1: get_governed_catalog
# ==============================================================================
@mcp_server.tool()
def get_governed_catalog() -> Dict[str, Any]:
    """
    Inspect database tables, columns, foreign keys, and allowed metrics authorized for this server's role.
    Always call this first to discover valid tables and column names before writing queries.
    """
    policy = ACTIVE_POLICY
    allowed_tables = {t.lower() for t in policy.allowed_tables}

    governed_tables = {
        tbl: schema
        for tbl, schema in BASE_CATALOG.items()
        if tbl.lower() in allowed_tables
    }

    foreign_key_relationships = [
        "orders.customer_id = customers.customer_id",
        "orders.order_id = order_payments.order_id",
        "orders.order_id = order_items.order_id",
        "order_items.product_id = products.product_id",
        "orders.order_id = order_reviews.order_id",
    ]

    return {
        "role": policy.role_name,
        "max_query_limit": policy.max_limit,
        "row_level_security_enforced": policy.rls_predicate is not None,
        "rls_predicate": policy.rls_predicate,
        "authorized_tables": governed_tables,
        "relationships": foreign_key_relationships,
        "guidelines": (
            "1. Tables and columns outside this list are forbidden and will be rejected by the AST security guard. "
            "2. Direct selection or filtering on customer tax IDs, phones, or credit cards is strictly blocked. "
            "3. Use resolve_entity_literals to verify state abbreviations and category names before querying. "
            "4. Each aggregation may use at most one of order_payments, order_items, order_reviews."
        ),
    }


# ==============================================================================
# MCP Tool 2: resolve_entity_literals
# ==============================================================================
@mcp_server.tool()
def resolve_entity_literals(terms: List[str]) -> Dict[str, Any]:
    """
    Resolve conversational entity terms (e.g. 'Rio de Janeiro', 'credit card', 'watches')
    to verified database enum values before generating SQL.
    Prevents SQL syntax and zero-row matching errors.

    Args:
        terms: List of entity strings mentioned by the user to ground.
    """
    grounder = get_grounder()
    results = {}
    for term in terms:
        resolved = grounder.resolve_term(term)
        if resolved:
            results[term] = {
                "grounded_value": resolved.canonical_literal,
                "domain": resolved.domain,
                "confidence": round(resolved.confidence, 4),
                "sql_predicate_hint": f"{resolved.domain} = '{resolved.canonical_literal}'",
            }
        else:
            results[term] = {
                "grounded_value": None,
                "domain": None,
                "confidence": 0.0,
                "hint": "No exact or fuzzy match found. Query column directly or use LIKE.",
            }

    return {
        "grounded_count": sum(1 for v in results.values() if v["grounded_value"] is not None),
        "entities": results,
    }


# ==============================================================================
# MCP Tool 3: execute_hybrid_query
# ==============================================================================
@mcp_server.tool()
def execute_hybrid_query(
    sql_filter: str = "1=1",
    semantic_query: str = "",
    bm25_keywords: Optional[str] = None,
    aggregations: Optional[List[str]] = None,
    limit: int = 5,
    match_top_percent: float = 1.0,
) -> Dict[str, Any]:
    """
    Execute a compound hybrid query joining structured SQL tables with unstructured review text.
    Enforces 5-Gate AST security, executes Plan A/B pushdown, and returns local PII-scrubbed evidence.

    Args:
        sql_filter: Relational SQL WHERE clause (e.g. "customers.customer_state = 'RJ' AND order_payments.payment_value > 300").
        semantic_query: Natural language complaint or concept to search for in customer reviews.
        bm25_keywords: Optional specific keywords for lexical BM25 matching (auto-extracted if omitted).
        aggregations: SQL aggregations computed once per distinct order (e.g. ["COUNT(*)", "SUM(order_payments.payment_value)"]).
            Each one may use at most one of order_payments, order_items, order_reviews.
        limit: Number of top cited review evidence snippets to return (max 10).
        match_top_percent: A review counts as matching semantic_query if it is among this % of all
            reviews most similar to it (default 1.0). Lower is stricter. Drives aggregates.matched_scope.
    """
    if not semantic_query or not semantic_query.strip():
        return {
            "status": "error",
            "error_type": "INVALID_ARGUMENT",
            "message": "semantic_query must be provided for hybrid retrieval.",
        }

    policy = ACTIVE_POLICY
    capped_limit = min(max(1, limit), 10)
    target_aggregations = aggregations or [
        "COUNT(DISTINCT orders.order_id) AS total_orders",
        "ROUND(SUM(order_payments.payment_value), 2) AS total_amount",
    ]

    request = HybridQueryRequest(
        sql_filter=sql_filter,
        semantic_query=semantic_query,
        bm25_keywords=bm25_keywords,
        aggregations=target_aggregations,
        limit=capped_limit,
        match_top_percent=min(max(0.01, match_top_percent), 100.0),
    )

    executor = get_executor()

    try:
        res = executor.execute(request, policy=policy)

        formatted_citations = [
            {
                "citation_rank": i + 1,
                "order_id": c.order_id,
                "review_id": c.review_id,
                "review_score": f"{c.review_score}/5",
                "similarity_or_distance": round(c.score_or_distance, 4),
                "review_snippet": c.review_comment,
            }
            for i, c in enumerate(res.citations)
        ]

        return {
            "status": "success",
            "plan_type": res.plan_type,
            "estimated_candidates": res.estimated_candidates,
            "telemetry": {
                "probe_latency_ms": res.probe_latency_ms,
                "engine_latency_ms": res.engine_latency_ms,
                "total_turn_ms": res.total_latency_ms,
            },
            "aggregates": {
                "filter_scope": {
                    "description": "Every order matching sql_filter",
                    "orders": res.estimated_candidates,
                    "metrics": {k: str(v) for k, v in res.aggregates.items()},
                },
                "matched_scope": {
                    "description": f"Orders matching sql_filter with a review in the top {request.match_top_percent}% most similar to semantic_query",
                    "orders": res.matched_orders,
                    "metrics": {k: str(v) for k, v in res.matched_aggregates.items()},
                },
            },
            "citations_returned": len(formatted_citations),
            "citations": formatted_citations,
            "security_governance": {
                "active_role": policy.role_name,
                "ast_validated": True,
                "pii_scrubbed": True,
                "hmac_tokenized": True,
            },
        }

    except ASTSecurityViolation as sec_err:
        logger.warning(f"Security Gate {sec_err.gate} blocked query: {sec_err}")
        return {
            "status": "security_violation",
            "gate": sec_err.gate,
            "violation_type": sec_err.violation_type,
            "message": str(sec_err),
            "guidance": (
                "Your SQL violated zero-trust security policies. Check get_governed_catalog "
                "for permitted tables/columns and ensure only read-only SELECT filters are used."
            ),
        }
    except ValueError as e:
        # Our own validation messages (e.g. a metric that would double count) are safe and actionable.
        return {"status": "execution_error", "error_type": "ValueError", "message": str(e)}
    except Exception as e:
        # Full details stay in the server log; the model only gets the first line (no SQL text or paths).
        logger.error(f"Execution failed: {e}", exc_info=True)
        return {
            "status": "execution_error",
            "error_type": type(e).__name__,
            "message": str(e).splitlines()[0][:200],
        }


def run_server(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8000,
    role: str = "analyst",
) -> None:
    """Start DualQuery MCP Server over stdio, sse, or streamable-http with a fixed role."""
    global ACTIVE_POLICY
    if role not in ROLE_POLICIES:
        raise ValueError(f"Unknown role '{role}'. Choose one of {sorted(ROLE_POLICIES)}.")
    ACTIVE_POLICY = ROLE_POLICIES[role]
    # stdout carries the stdio JSON-RPC stream, so logs must go to stderr.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if transport == "stdio":
        logger.info("Starting DualQuery MCP Server (stdio transport)...")
        mcp_server.run(transport="stdio")
    else:
        logger.info(f"Starting DualQuery MCP Server ({transport} transport on {host}:{port})...")
        mcp_server.run(transport=transport, host=host, port=port)  # type: ignore[arg-type]


def main() -> None:
    """CLI entrypoint for `dualquery-mcp` supporting stdio, sse, and streamable-http transports."""
    import argparse

    parser = argparse.ArgumentParser(description="Start DualQuery Zero-Trust MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind host for SSE/HTTP transports")
    parser.add_argument("--port", type=int, default=8000, help="Bind port for SSE/HTTP transports")
    parser.add_argument("--role", choices=sorted(ROLE_POLICIES), default="analyst", help="Security role for every tool call")
    args = parser.parse_args()
    run_server(transport=args.transport, host=args.host, port=args.port, role=args.role)


if __name__ == "__main__":
    main()
