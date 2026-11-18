"""
DualQuery CLI Interface.
Provides command-line commands for Bring-Your-Own-Export (BYOE) ingestion,
financial leakage audits, and running the Zero-Trust MCP Server.
"""

import sys
import argparse
from pathlib import Path
from rich.console import Console

from dualquery.config import settings
from dualquery.ingestion.byoe_adapter import BYOEAdapter
from dualquery.engine.audit_runner import CommercialAuditRunner

console = Console()

OLIST_FILES = [
    "olist_customers_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_products_dataset.csv",
    "product_category_name_translation.csv",
]


def handle_ingest(args: argparse.Namespace) -> None:
    """Handle 'dualquery ingest' command."""
    tx_path = Path(args.transactions)
    fb_path = Path(args.feedback)
    db_path = Path(args.db) if args.db else settings.DATA_DIR / "byoe_audit.duckdb"

    if not tx_path.exists():
        console.print(f"[bold red]Error:[/bold red] Transactions file not found: {tx_path}")
        sys.exit(1)
    if not fb_path.exists():
        console.print(f"[bold red]Error:[/bold red] Feedback file not found: {fb_path}")
        sys.exit(1)

    console.print(f"[bold cyan]Starting BYOE Ingestion into[/bold cyan] {db_path}...")
    adapter = BYOEAdapter(db_path=db_path)
    report = adapter.ingest(tx_path, fb_path)

    console.print(f"[bold green]Ingestion complete in {report.duration_seconds}s![/bold green]")
    console.print(f"  • Transactions ingested: {report.transactions_ingested:,}")
    console.print(f"  • Feedback notes ingested & embedded: {report.feedback_notes_ingested:,}")
    console.print(f"  • Mapped Transaction Columns: {report.column_mappings['transactions']}")
    console.print(f"  • Mapped Feedback Columns: {report.column_mappings['feedback']}")


def handle_ingest_olist(args: argparse.Namespace) -> None:
    """Handle 'dualquery ingest-olist': build the Olist DuckDB the MCP server queries."""
    from dualquery.ingestion.olist_loader import OlistLoader
    from dualquery.ingestion.literal_indexer import LiteralIndexer

    data_dir = Path(args.data_dir)
    db_path = Path(args.db) if args.db else settings.DUCKDB_PATH
    missing = [f for f in OLIST_FILES if not (data_dir / f).exists()]
    if missing:
        console.print(f"[bold red]Error:[/bold red] Missing Olist CSVs in {data_dir}: {missing}")
        console.print("Download them from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce")
        sys.exit(1)

    OlistLoader(db_path=db_path, data_dir=data_dir).run(max_reviews=args.max_reviews)
    LiteralIndexer(db_path=db_path).build_index()


def handle_audit(args: argparse.Namespace) -> None:
    """Handle 'dualquery audit' command."""
    db_path = Path(args.db) if args.db else settings.DATA_DIR / "byoe_audit.duckdb"

    # Optional Ingestion step if CSVs are supplied and DB doesn't exist or not skipping
    if not args.skip_ingest and args.transactions and args.feedback:
        tx_path = Path(args.transactions)
        fb_path = Path(args.feedback)
        if tx_path.exists() and fb_path.exists():
            console.print(f"[bold cyan]Ingesting fresh CSV data into[/bold cyan] {db_path}...")
            adapter = BYOEAdapter(db_path=db_path)
            adapter.ingest(tx_path, fb_path)

    if not db_path.exists():
        console.print(f"[bold red]Error:[/bold red] Database not found at {db_path}. Please supply --transactions and --feedback or run 'dualquery ingest'.")
        sys.exit(1)

    runner = CommercialAuditRunner(db_path=db_path)
    report = runner.run_audit()
    runner.render_dashboard(report, console=console)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        console.print(f"[dim]Saved JSON audit report to {out_p}[/dim]")


def handle_serve(args: argparse.Namespace) -> None:
    """Handle 'dualquery serve' command."""
    from dualquery.mcp.server import run_server
    run_server(transport=args.transport, host=args.host, port=args.port, role=args.role)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="dualquery",
        description="DualQuery: Zero-Trust Hybrid Relational-Semantic Query Engine & MCP Server",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest arbitrary CSV exports into DuckDB")
    ingest_parser.add_argument("-t", "--transactions", required=True, help="Path to transactions CSV")
    ingest_parser.add_argument("-f", "--feedback", required=True, help="Path to return notes / feedback CSV")
    ingest_parser.add_argument("--db", default=None, help="Custom DuckDB database path")
    ingest_parser.set_defaults(func=handle_ingest)

    # Olist ingest command (builds the database behind the MCP server)
    olist_parser = subparsers.add_parser("ingest-olist", help="Build olist.duckdb (tables, embeddings, HNSW + BM25) for the MCP server")
    olist_parser.add_argument("--data-dir", default=str(settings.DATA_DIR / "olist_dataset"), help="Folder with the Kaggle Olist CSVs")
    olist_parser.add_argument("--db", default=None, help="Output DuckDB path (default: data/olist.duckdb or DUALQUERY_DB_PATH)")
    olist_parser.add_argument("--max-reviews", type=int, default=None, help="Load only the first N reviews (quick demo build)")
    olist_parser.set_defaults(func=handle_ingest_olist)

    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Run commercial financial leakage audit")
    audit_parser.add_argument("-t", "--transactions", default=None, help="Path to transactions CSV (optional if already ingested)")
    audit_parser.add_argument("-f", "--feedback", default=None, help="Path to feedback CSV (optional if already ingested)")
    audit_parser.add_argument("--db", default=None, help="Custom DuckDB database path")
    audit_parser.add_argument("-o", "--output-json", default=None, help="Path to export audit report as JSON")
    audit_parser.add_argument("--skip-ingest", action="store_true", help="Skip re-ingestion and audit existing database")
    audit_parser.set_defaults(func=handle_audit)

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start the JSON-RPC 2.0 Zero-Trust MCP Server")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="MCP transport protocol (default: stdio)",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host for SSE/HTTP transports")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port for SSE/HTTP transports")
    serve_parser.add_argument(
        "--role",
        choices=["analyst", "seller_restricted", "support_agent"],
        default="analyst",
        help="Security role applied to every tool call (fixed for the server's lifetime)",
    )
    serve_parser.set_defaults(func=handle_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

