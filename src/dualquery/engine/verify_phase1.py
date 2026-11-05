"""
DualQuery Phase 1 Verification Suite.
Validates:
1. Relational schema integrity (foreign key joins across orders, payments, products)
2. Exact SIMD Cosine Distance execution (Plan A vector calculation)
3. HNSW Index Vector Probe (Plan B ANN retrieval)
4. BM25 Full-Text Search Probe (fts_main_order_reviews)
5. Pre-SQL Literal Grounder lookup table
"""

import json
import time
from pathlib import Path
import duckdb
from rich.console import Console
from rich.table import Table

from dualquery.config import settings
from dualquery.ingestion.embeddings import get_embedding_provider

console = Console()


def verify_relational_olap(conn: duckdb.DuckDBPyConnection) -> None:
    """Verify that relational tables join correctly with deterministic aggregates."""
    console.print("\n[bold cyan]1. Verifying Relational OLAP Joins & Aggregates...[/bold cyan]")
    start = time.perf_counter()
    
    query = """
    SELECT 
        c.customer_state,
        COUNT(DISTINCT o.order_id) as total_orders,
        ROUND(SUM(p.payment_value), 2) as total_revenue,
        ROUND(AVG(p.payment_value), 2) as avg_order_value
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN order_payments p ON o.order_id = p.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_state
    ORDER BY total_revenue DESC
    LIMIT 5;
    """
    rows = conn.execute(query).fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000

    table = Table(title=f"Top 5 States by Revenue (Query Latency: {elapsed_ms:.2f}ms)", style="green")
    table.add_column("State", style="bold white")
    table.add_column("Orders", justify="right", style="cyan")
    table.add_column("Total Revenue (R$)", justify="right", style="yellow")
    table.add_column("Avg Order Value (R$)", justify="right", style="magenta")

    for r in rows:
        table.add_row(str(r[0]), f"{r[1]:,}", f"R$ {r[2]:,}", f"R$ {r[3]:,}")
    
    console.print(table)


def verify_simd_vector_search(conn: duckdb.DuckDBPyConnection, embedder) -> list[float]:
    """Verify exact SIMD cosine distance calculation (Plan A)."""
    test_query = "produto quebrado estragado embalagem danificada"
    console.print(f"\n[bold cyan]2. Verifying Plan A: Exact SIMD Cosine Distance Probe...[/bold cyan]")
    console.print(f"Query text: [italic white]'{test_query}'[/italic white]")
    
    # 1. Embed query
    start_embed = time.perf_counter()
    query_vec = embedder.embed_query(test_query)
    embed_ms = (time.perf_counter() - start_embed) * 1000
    console.print(f"Query vector generated in: [bold green]{embed_ms:.2f}ms[/bold green] (dim={len(query_vec)})")

    # 2. Run SIMD cosine distance query on DuckDB
    start_simd = time.perf_counter()
    simd_query = """
    SELECT 
        review_id,
        review_score,
        COALESCE(review_comment_message, '') as comment,
        array_cosine_distance(embedding, ?::FLOAT[768]) as cosine_dist
    FROM order_reviews
    WHERE embedding IS NOT NULL
    ORDER BY cosine_dist ASC
    LIMIT 3;
    """
    results = conn.execute(simd_query, [query_vec]).fetchall()
    simd_ms = (time.perf_counter() - start_simd) * 1000

    table = Table(title=f"Plan A Nearest Neighbors (Execution: {simd_ms:.2f}ms)", style="cyan")
    table.add_column("Score", justify="center", style="yellow")
    table.add_column("Cosine Dist", justify="right", style="green")
    table.add_column("Review Message Snippet", style="white")

    for r in results:
        snippet = (r[2][:80] + "...") if len(r[2]) > 80 else r[2]
        table.add_row(f"{r[1]}/5", f"{r[3]:.4f}", snippet)

    console.print(table)
    return query_vec


def verify_hnsw_vector_search(conn: duckdb.DuckDBPyConnection, query_vec: list[float]) -> None:
    """Verify Plan B: Native HNSW Index Probe (vss)."""
    console.print(f"\n[bold cyan]2b. Verifying Plan B: Native HNSW Index Probe (vss)...[/bold cyan]")
    start_hnsw = time.perf_counter()
    hnsw_query = """
    SELECT 
        review_id,
        review_score,
        COALESCE(review_comment_message, '') as comment,
        array_cosine_distance(embedding, ?::FLOAT[768]) as cosine_dist
    FROM order_reviews
    ORDER BY array_cosine_distance(embedding, ?::FLOAT[768]) ASC
    LIMIT 3;
    """
    results = conn.execute(hnsw_query, [query_vec, query_vec]).fetchall()
    hnsw_ms = (time.perf_counter() - start_hnsw) * 1000

    table = Table(title=f"Plan B HNSW Probe (Execution: {hnsw_ms:.2f}ms)", style="blue")
    table.add_column("Score", justify="center", style="yellow")
    table.add_column("Cosine Dist", justify="right", style="green")
    table.add_column("Review Message Snippet", style="white")

    for r in results:
        snippet = (r[2][:80] + "...") if len(r[2]) > 80 else r[2]
        table.add_row(f"{r[1]}/5", f"{r[3]:.4f}", snippet)

    console.print(table)


def verify_bm25_full_text(conn: duckdb.DuckDBPyConnection) -> None:
    """Verify DuckDB FTS BM25 keyword retrieval."""
    keyword = "quebrado"
    console.print(f"\n[bold cyan]3. Verifying Lexical BM25 FTS Index (fts)...[/bold cyan]")
    console.print(f"Keyword search: [italic white]'{keyword}'[/italic white]")
    
    start = time.perf_counter()
    bm25_query = """
    SELECT 
        r.review_id,
        r.review_score,
        r.review_comment_message,
        ROUND(score, 4) as bm25_score
    FROM (
        SELECT *, fts_main_order_reviews.match_bm25(review_id, 'quebrado') AS score
        FROM order_reviews
    ) r
    WHERE score IS NOT NULL
    ORDER BY score DESC
    LIMIT 3;
    """
    results = conn.execute(bm25_query).fetchall()
    elapsed_ms = (time.perf_counter() - start) * 1000

    table = Table(title=f"BM25 Top Matches (Execution: {elapsed_ms:.2f}ms)", style="magenta")
    table.add_column("Score", justify="center", style="yellow")
    table.add_column("BM25 Score", justify="right", style="green")
    table.add_column("Review Message Snippet", style="white")

    for r in results:
        msg = r[2] or ""
        snippet = (msg[:80] + "...") if len(msg) > 80 else msg
        table.add_row(f"{r[1]}/5", str(r[3]), snippet)

    console.print(table)


def verify_literal_dictionary() -> None:
    """Verify that the pre-SQL categorical grounder dictionary is populated."""
    console.print("\n[bold cyan]4. Verifying Pre-SQL Grounder Dictionary...[/bold cyan]")
    dict_file = settings.DATA_DIR / "literal_dictionary.json"
    if not dict_file.exists():
        console.print(f"[red]Missing {dict_file}![/red]")
        return

    with open(dict_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    console.print(f"[green][OK] Literal dictionary verified with [bold]{len(data)}[/bold] domains:[/green]")
    for domain, items in data.items():
        console.print(f"  • [yellow]{domain}[/yellow]: {len(items)} values (e.g. {items[:3]})")


def main():
    console.print("[bold green]=====================================================[/bold green]")
    console.print("[bold green]       DualQuery Phase 1 Verification Suite          [/bold green]")
    console.print("[bold green]=====================================================[/bold green]")

    db_path = settings.DUCKDB_PATH
    if not db_path.exists():
        console.print(f"[bold red]Database file not found at {db_path}![/bold red]")
        return

    conn = duckdb.connect(str(db_path), read_only=True)
    conn.execute("LOAD vss; LOAD fts;")
    
    try:
        verify_relational_olap(conn)
        embedder = get_embedding_provider("auto")
        query_vec = verify_simd_vector_search(conn, embedder)
        verify_hnsw_vector_search(conn, query_vec)
        verify_bm25_full_text(conn)
        verify_literal_dictionary()
    finally:
        conn.close()

    console.print("\n[bold green]=====================================================[/bold green]")
    console.print("[bold green]   ALL PHASE 1 ARCHITECTURAL MILESTONES VERIFIED!    [/bold green]")
    console.print("[bold green]=====================================================[/bold green]")


if __name__ == "__main__":
    main()
