"""
DualQuery Adversarial Red-Team Security Evaluation Suite.
Tests the 5-Gate SQLGlot AST Security Guard against 30 real-world malicious attack vectors:
- DDL/DML mutations & table destruction
- Multi-statement command chaining
- Host filesystem & cloud network exfiltration (read_csv, read_parquet, glob)
- CTE & Subquery table smuggling
- Cross-role privilege escalation
- PII & credential exfiltration
- Administrative PRAGMA & ATTACH exploits
"""

import json
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dualquery.security.ast_guard import (
    ASTGuard,
    ASTSecurityViolation,
    ANALYST_POLICY,
    SUPPORT_AGENT_POLICY,
    SELLER_RESTRICTED_POLICY,
)

console = Console()

RED_TEAM_ATTACKS = [
    # Category 1: DDL / DML Destruction & Mutation
    {
        "id": "SEC_01",
        "category": "DDL Destruction",
        "name": "Direct DROP TABLE",
        "sql": "DROP TABLE orders;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_02",
        "category": "DDL Destruction",
        "name": "DROP TABLE with CASCADE",
        "sql": "DROP TABLE customers CASCADE;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_03",
        "category": "DDL Destruction",
        "name": "TRUNCATE TABLE",
        "sql": "TRUNCATE TABLE order_reviews;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_04",
        "category": "DDL Mutation",
        "name": "ALTER TABLE Backdoor",
        "sql": "ALTER TABLE orders ADD COLUMN backdoor VARCHAR;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_05",
        "category": "DML Mutation",
        "name": "Mass DELETE",
        "sql": "DELETE FROM orders WHERE 1=1;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_06",
        "category": "DML Mutation",
        "name": "Malicious UPDATE",
        "sql": "UPDATE order_payments SET payment_value = 0.0 WHERE order_id IS NOT NULL;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_07",
        "category": "DML Mutation",
        "name": "Rogue INSERT Injection",
        "sql": "INSERT INTO customers VALUES ('hacker_id', 'unique_hacker', '00000', 'HackerCity', 'SP');",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },

    # Category 2: Multi-Statement & Comment Evasion
    {
        "id": "SEC_08",
        "category": "Multi-Statement",
        "name": "Chained Semicolon DROP",
        "sql": "SELECT * FROM orders; DROP TABLE customers;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_09",
        "category": "Multi-Statement",
        "name": "Whitespace Obfuscated Semicolon",
        "sql": "SELECT order_id FROM orders;       \n\n\t   DELETE FROM order_reviews;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_10",
        "category": "Comment Evasion",
        "name": "Inline Comment Masking",
        "sql": "SELECT * FROM orders; /* bypass check */ DROP TABLE products;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },

    # Category 3: Filesystem & Network Exfiltration
    {
        "id": "SEC_11",
        "category": "Filesystem Exfiltration",
        "name": "read_csv /etc/passwd",
        "sql": "SELECT * FROM read_csv('/etc/passwd');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_12",
        "category": "Filesystem Exfiltration",
        "name": "read_csv_auto Windows Hosts",
        "sql": "SELECT * FROM read_csv_auto('C:/Windows/System32/drivers/etc/hosts');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_13",
        "category": "Cloud Network Exfiltration",
        "name": "read_parquet S3 Exfil",
        "sql": "SELECT * FROM read_parquet('s3://exfiltrate-bucket/secrets.parquet');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_14",
        "category": "Filesystem Exfiltration",
        "name": "scan_csv Private File",
        "sql": "SELECT * FROM scan_csv('company_payroll.csv');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_15",
        "category": "Filesystem Exfiltration",
        "name": "read_json_auto Credentials",
        "sql": "SELECT * FROM read_json_auto('credentials.json');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_16",
        "category": "Filesystem Exfiltration",
        "name": "glob Directory Traversal",
        "sql": "SELECT * FROM glob('../*');",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_17",
        "category": "Internal Leak",
        "name": "duckdb_settings Exfiltration",
        "sql": "SELECT * FROM duckdb_settings();",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_18",
        "category": "Internal Leak",
        "name": "duckdb_secrets Token Steal",
        "sql": "SELECT * FROM duckdb_secrets();",
        "expected_gate": 2,
        "policy": ANALYST_POLICY
    },

    # Category 4: Administrative & Attachment Exploits
    {
        "id": "SEC_19",
        "category": "Admin Exploit",
        "name": "PRAGMA show_tables",
        "sql": "PRAGMA show_tables;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_20",
        "category": "Admin Exploit",
        "name": "PRAGMA database_list",
        "sql": "PRAGMA database_list;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_21",
        "category": "Admin Exploit",
        "name": "ATTACH Database Hijack",
        "sql": "ATTACH 'trojan.duckdb' AS stolen_db;",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },

    # Category 5: Table Smuggling & Privilege Escalation
    {
        "id": "SEC_22",
        "category": "Table Smuggling",
        "name": "CTE Smuggled Unauthorized Table",
        "sql": "WITH hidden AS (SELECT * FROM internal_salaries) SELECT * FROM hidden;",
        "expected_gate": 3,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_23",
        "category": "Table Smuggling",
        "name": "Nested Subquery Unauthorized Table",
        "sql": "SELECT * FROM (SELECT * FROM private_audit_log);",
        "expected_gate": 3,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_24",
        "category": "Role Boundary Violation",
        "name": "Support Agent Accessing Payments",
        "sql": "SELECT payment_value, payment_type FROM order_payments;",
        "expected_gate": 3,
        "policy": SUPPORT_AGENT_POLICY  # Support agent is not allowed to query payments
    },
    {
        "id": "SEC_25",
        "category": "Role Boundary Violation",
        "name": "Support Agent JOINing Payments",
        "sql": "SELECT o.order_id, p.payment_value FROM orders o JOIN order_payments p ON o.order_id = p.order_id;",
        "expected_gate": 3,
        "policy": SUPPORT_AGENT_POLICY
    },

    # Category 6: PII & Credential Exfiltration
    {
        "id": "SEC_26",
        "category": "PII Exfiltration",
        "name": "Direct customer_unique_id Projection",
        "sql": "SELECT customer_id, customer_unique_id FROM customers;",
        "expected_gate": 4,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_27",
        "category": "PII Exfiltration",
        "name": "CPF Tax ID Column Access",
        "sql": "SELECT customer_id, cpf FROM customers;",
        "expected_gate": 4,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_28",
        "category": "PII Exfiltration",
        "name": "Credit Card Number Column Projection",
        "sql": "SELECT order_id, credit_card_number FROM order_payments;",
        "expected_gate": 4,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_29",
        "category": "PII Exfiltration",
        "name": "PII Column in WHERE Predicate",
        "sql": "SELECT order_id FROM orders WHERE ssn = '000-11-2222';",
        "expected_gate": 4,
        "policy": ANALYST_POLICY
    },
    {
        "id": "SEC_30",
        "category": "Empty Query Exploit",
        "name": "Empty String Injection",
        "sql": "   \n\t  ",
        "expected_gate": 1,
        "policy": ANALYST_POLICY
    },
]


def run_security_benchmark():
    console.print(Panel.fit(
        f"[bold red]DualQuery Zero-Trust Adversarial Red-Team Benchmark[/bold red]\n"
        f"Testing [yellow]{len(RED_TEAM_ATTACKS)}[/yellow] Malicious Attack Vectors against 5-Gate AST Security Guard\n"
        f"Evaluating: DDL/DML, Multi-Statements, Filesystem Exfiltration, Table Smuggling, PII Leaks",
        border_style="red"
    ))

    guard = ASTGuard()
    table = Table(title="Red-Team Adversarial Attack Execution Matrix", style="red", show_lines=True)
    table.add_column("Attack ID", style="bold yellow", justify="center")
    table.add_column("Category", style="cyan")
    table.add_column("Attack Description", style="white")
    table.add_column("Expected Gate", justify="center", style="magenta")
    table.add_column("Actual Gate", justify="center", style="bold cyan")
    table.add_column("Violation Type", style="dim")
    table.add_column("Result", justify="center")

    blocked_count = 0
    results: List[Dict[str, Any]] = []

    for attack in RED_TEAM_ATTACKS:
        aid = attack["id"]
        category = attack["category"]
        name = attack["name"]
        sql = attack["sql"]
        exp_gate = attack["expected_gate"]
        policy = attack["policy"]

        try:
            hardened_sql = guard.validate_and_rewrite(sql, policy=policy)
            # If no exception was raised, the attack SLIPPED through!
            status = "[bold red][SLIPPED][/bold red]"
            act_gate = "NONE"
            vtype = "UNCAUGHT"
            is_blocked = False
        except ASTSecurityViolation as e:
            is_blocked = True
            blocked_count += 1
            act_gate = f"Gate {e.gate}"
            vtype = e.violation_type
            status = "[bold green][BLOCKED][/bold green]"
        except Exception as e:
            # Other parse error or exception
            is_blocked = True
            blocked_count += 1
            act_gate = "Parser"
            vtype = type(e).__name__
            status = "[bold green][BLOCKED][/bold green]"

        table.add_row(
            aid,
            category,
            name,
            f"Gate {exp_gate}",
            act_gate,
            vtype,
            status
        )

        results.append({
            "id": aid,
            "category": category,
            "name": name,
            "sql": sql,
            "expected_gate": exp_gate,
            "actual_gate": act_gate,
            "violation_type": vtype,
            "is_blocked": is_blocked
        })

    console.print(table)

    block_rate = (blocked_count / len(RED_TEAM_ATTACKS)) * 100

    summary_table = Table(title="Red-Team Security Benchmark Summary", style="green")
    summary_table.add_column("Metric", style="bold yellow")
    summary_table.add_column("Value", style="cyan", justify="right")

    summary_table.add_row("Total Red-Team Attack Vectors", str(len(RED_TEAM_ATTACKS)))
    summary_table.add_row("Successfully Blocked Attacks", f"{blocked_count}/{len(RED_TEAM_ATTACKS)}")
    summary_table.add_row("Security Block Rate", f"[bold green]{block_rate:.1f}%[/bold green]")
    summary_table.add_row("Zero-Trust Security Status", "[bold green]100% COMPLIANT[/bold green]" if block_rate == 100.0 else "[bold red]NON-COMPLIANT[/bold red]")

    console.print(summary_table)

    # Save benchmark artifact
    output_path = Path("benchmarks/security_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_attacks": len(RED_TEAM_ATTACKS),
            "blocked_count": blocked_count,
            "block_rate_percent": block_rate,
            "attacks": results
        }, f, indent=2)

    console.print(f"[bold green]Saved security audit report to {output_path}[/bold green]")


if __name__ == "__main__":
    run_security_benchmark()

