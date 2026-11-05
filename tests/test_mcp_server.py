"""
Test suite for DualQuery MCP Server tools.
Verifies tool discovery, role RBAC catalog, literal grounding, hybrid execution, and security gating.
"""

import asyncio
import pytest
from rich.console import Console
from dualquery.mcp.server import mcp_server

console = Console()


async def run_tests():
    console.print("[bold cyan]=== DualQuery MCP Server Verification Suite ===[/bold cyan]\n")

    # 1. List registered tools
    tools = await mcp_server.list_tools()
    tool_names = [t.name for t in tools]
    console.print(f"[bold green][OK][/bold green] Registered Tools: {tool_names}")
    assert "get_governed_catalog" in tool_names
    assert "resolve_entity_literals" in tool_names
    assert "execute_hybrid_query" in tool_names

    # 2. Test get_governed_catalog RBAC
    console.print("\n[bold yellow]Testing Tool 1: get_governed_catalog (Role RBAC)...[/bold yellow]")
    analyst_cat = await mcp_server.call_tool("get_governed_catalog", {"role": "analyst"})
    support_cat = await mcp_server.call_tool("get_governed_catalog", {"role": "support_agent"})

    analyst_tables = list(analyst_cat.structured_content["result"]["authorized_tables"].keys())
    support_tables = list(support_cat.structured_content["result"]["authorized_tables"].keys())
    console.print(f"Analyst tables ({len(analyst_tables)}): {analyst_tables}")
    console.print(f"Support Agent tables ({len(support_tables)}): {support_tables}")
    assert "order_payments" in analyst_tables
    assert "order_payments" not in support_tables
    console.print("[bold green][OK][/bold green] Role-governed catalog filtering verified!")

    # 3. Test resolve_entity_literals
    console.print("\n[bold yellow]Testing Tool 2: resolve_entity_literals...[/bold yellow]")
    ground_res = await mcp_server.call_tool(
        "resolve_entity_literals",
        {"terms": ["Rio de Janeiro", "credit card", "relogios"]}
    )
    entities = ground_res.structured_content["result"]["entities"]
    for term, data in entities.items():
        console.print(f" - '{term}' -> [bold green]{data['grounded_value']}[/bold green] ({data['domain']}, conf: {data['confidence']})")
    assert entities["Rio de Janeiro"]["grounded_value"] == "RJ"
    assert entities["credit card"]["grounded_value"] == "credit_card"
    console.print("[bold green][OK][/bold green] Entity literal grounding verified!")

    # 4. Test execute_hybrid_query (Valid execution)
    console.print("\n[bold yellow]Testing Tool 3: execute_hybrid_query (Compound Query & PII Redaction)...[/bold yellow]")
    q_res = await mcp_server.call_tool(
        "execute_hybrid_query",
        {
            "sql_filter": "customers.customer_state = 'RJ' AND order_payments.payment_type = 'credit_card' AND order_payments.payment_value > 300",
            "semantic_query": "produto quebrado danificado embalagem veio em pedacos",
            "limit": 2,
            "role": "analyst"
        }
    )
    res_data = q_res.structured_content["result"]
    console.print(f"Status: [bold green]{res_data['status']}[/bold green]")
    console.print(f"Plan Chosen: [cyan]{res_data['plan_type']}[/cyan] (Candidates: {res_data['estimated_candidates']})")
    console.print(f"Aggregates: {res_data['aggregates']}")
    console.print(f"Telemetry: {res_data['telemetry']}")
    console.print(f"Citations: {res_data['citations_returned']} returned")
    for c in res_data["citations"]:
        console.print(f" - Citation #{c['citation_rank']}: Order {c['order_id']} | Score: {c['review_score']} | Snippet: {c['review_snippet'][:70]}...")
    assert res_data["status"] == "success"
    assert len(res_data["citations"]) > 0
    assert "[ORD_" in res_data["citations"][0]["order_id"]
    console.print("[bold green][OK][/bold green] Hybrid query executed with HMAC tokenization and PII scrubbing!")

    # 5. Test execute_hybrid_query (Malicious attack blocked)
    console.print("\n[bold yellow]Testing Security Attack over MCP (Chained DROP TABLE)...[/bold yellow]")
    sec_res = await mcp_server.call_tool(
        "execute_hybrid_query",
        {
            "sql_filter": "1=1; DROP TABLE customers;",
            "semantic_query": "broken box",
            "role": "analyst"
        }
    )
    sec_data = sec_res.structured_content["result"]
    console.print(f"Status: [bold red]{sec_data['status']}[/bold red]")
    console.print(f"Gate: {sec_data.get('gate')} ({sec_data.get('violation_type')})")
    console.print(f"Message: {sec_data.get('message')}")
    assert sec_data["status"] == "security_violation"
    assert sec_data["gate"] == 1
    console.print("[bold green][OK][/bold green] Malicious SQL blocked by Gate 1 over MCP!")

    # 6. Test execute_hybrid_query (Role RBAC boundary blocked)
    console.print("\n[bold yellow]Testing Role Boundary Violation over MCP...[/bold yellow]")
    role_sec_res = await mcp_server.call_tool(
        "execute_hybrid_query",
        {
            "sql_filter": "order_payments.payment_value > 500",
            "semantic_query": "expensive fee",
            "role": "support_agent"
        }
    )
    role_sec_data = role_sec_res.structured_content["result"]
    console.print(f"Status: [bold red]{role_sec_data['status']}[/bold red]")
    console.print(f"Gate: {role_sec_data.get('gate')} ({role_sec_data.get('violation_type')})")
    console.print(f"Message: {role_sec_data.get('message')}")
    assert role_sec_data["status"] == "security_violation"
    assert role_sec_data["gate"] == 3
    console.print("[bold green][OK][/bold green] Cross-role privilege escalation blocked by Gate 3 over MCP!")

    console.print("\n[bold green]ALL 6 MCP SERVER VERIFICATION TESTS PASSED SUCCESSFULLY![/bold green]")


@pytest.mark.asyncio
async def test_mcp_server_full_suite():
    await run_tests()


if __name__ == "__main__":
    asyncio.run(run_tests())

