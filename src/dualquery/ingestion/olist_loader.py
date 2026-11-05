"""
DualQuery Olist Ingestion Pipeline.
Loads 100k Brazilian E-Commerce relational records, computes 768-d review embeddings,
and compiles DuckDB HNSW and BM25 FTS indexes.
"""

import time
from pathlib import Path
from typing import Optional
import duckdb
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from dualquery.config import settings
from dualquery.ingestion.embeddings import EmbeddingProvider, get_embedding_provider

console = Console()


class OlistLoader:
    def __init__(
        self,
        db_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ):
        self.db_path = db_path or settings.DUCKDB_PATH
        self.data_dir = data_dir or (settings.DATA_DIR / "olist_dataset")
        self.sql_dir = settings.SQL_DIR
        self.embedder = embedding_provider or get_embedding_provider("auto")
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        """Connect to DuckDB and ensure required extensions are loaded."""
        conn = duckdb.connect(str(self.db_path))
        conn.execute("INSTALL vss; LOAD vss;")
        conn.execute("INSTALL fts; LOAD fts;")
        conn.execute("SET hnsw_enable_experimental_persistence = true;")
        return conn

    def initialize_schema(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Execute the DDL schema script to create relational tables and types."""
        schema_file = self.sql_dir / "01_init_schema.sql"
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found at {schema_file}")
        
        console.print(f"[cyan]Executing DDL schema from: [bold]{schema_file}[/bold][/cyan]")
        
        # Read schema and execute statements
        # Note: We create indexes AFTER bulk data insertion for maximum speed!
        with open(schema_file, "r", encoding="utf-8") as f:
            statements = f.read().split(";")
            
        for stmt in statements:
            # Strip SQL comments
            lines = [l for l in stmt.splitlines() if not l.strip().startswith("--")]
            cleaned = "\n".join(lines).strip()
            # Skip index creation during initial table creation; we build them after loading data
            if cleaned and not cleaned.startswith("CREATE INDEX") and not cleaned.startswith("PRAGMA create_fts_index"):
                conn.execute(cleaned)

    def load_relational_tables(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Stream CSV files directly into DuckDB relational tables using multi-threaded SIMD readers."""
        console.print("[yellow]Streaming relational CSV datasets into DuckDB...[/yellow]")
        start = time.perf_counter()

        # 1. Customers
        cust_csv = (self.data_dir / "olist_customers_dataset.csv").as_posix()
        conn.execute(f"""
            INSERT OR IGNORE INTO customers
            SELECT customer_id, customer_unique_id, customer_zip_code_prefix, customer_city, customer_state
            FROM read_csv_auto('{cust_csv}');
        """)

        # 2. Orders (includes virtual generated column delivery_delay_days)
        orders_csv = (self.data_dir / "olist_orders_dataset.csv").as_posix()
        conn.execute(f"""
            INSERT OR IGNORE INTO orders (
                order_id, customer_id, order_status, 
                order_purchase_timestamp, order_delivered_customer_date, order_estimated_delivery_date
            )
            SELECT 
                order_id, customer_id, order_status,
                TRY_CAST(order_purchase_timestamp AS TIMESTAMP),
                TRY_CAST(order_delivered_customer_date AS TIMESTAMP),
                TRY_CAST(order_estimated_delivery_date AS TIMESTAMP)
            FROM read_csv_auto('{orders_csv}');
        """)

        # 3. Order Payments
        payments_csv = (self.data_dir / "olist_order_payments_dataset.csv").as_posix()
        conn.execute(f"""
            INSERT OR IGNORE INTO order_payments
            SELECT order_id, payment_sequential, payment_type, payment_installments, payment_value
            FROM read_csv_auto('{payments_csv}');
        """)

        # 4. Products (Left join with category translations for English search)
        prod_csv = (self.data_dir / "olist_products_dataset.csv").as_posix()
        trans_csv = (self.data_dir / "product_category_name_translation.csv").as_posix()
        conn.execute(f"""
            INSERT OR IGNORE INTO products
            SELECT 
                p.product_id, 
                p.product_category_name, 
                COALESCE(t.product_category_name_english, p.product_category_name) AS product_category_name_english,
                TRY_CAST(p.product_weight_g AS INTEGER)
            FROM read_csv_auto('{prod_csv}') p
            LEFT JOIN read_csv_auto('{trans_csv}') t 
                ON p.product_category_name = t.product_category_name;
        """)

        # 5. Order Items
        items_csv = (self.data_dir / "olist_order_items_dataset.csv").as_posix()
        conn.execute(f"""
            INSERT OR IGNORE INTO order_items
            SELECT order_id, order_item_id, product_id, seller_id, price, freight_value
            FROM read_csv_auto('{items_csv}');
        """)

        elapsed = time.perf_counter() - start
        console.print(f"[green][OK] Relational tables populated in {elapsed:.2f}s[/green]")

    def load_and_embed_reviews(self, conn: duckdb.DuckDBPyConnection, max_reviews: Optional[int] = None) -> None:
        """
        Loads customer reviews, batches comments for semantic embedding on GPU/CPU,
        and persists vectors into order_reviews.
        """
        reviews_csv = (self.data_dir / "olist_order_reviews_dataset.csv").as_posix()
        console.print("[yellow]Ingesting raw review text...[/yellow]")

        # Insert base review text without embeddings first
        limit_clause = f"LIMIT {max_reviews}" if max_reviews else ""
        conn.execute(f"""
            INSERT OR IGNORE INTO order_reviews (
                review_id, order_id, review_score, review_comment_title, 
                review_comment_message, review_creation_date
            )
            SELECT 
                review_id, order_id, review_score, review_comment_title, 
                review_comment_message, TRY_CAST(review_creation_date AS TIMESTAMP)
            FROM read_csv_auto('{reviews_csv}')
            {limit_clause};
        """)

        # Fetch reviews that actually contain meaningful text to embed
        # (Filtering out null/empty reviews saves 60% of computation time!)
        cursor = conn.execute("""
            SELECT review_id, review_comment_title, review_comment_message
            FROM order_reviews
            WHERE review_comment_message IS NOT NULL 
              AND length(trim(review_comment_message)) > 0
              AND embedding IS NULL;
        """)
        rows_to_embed = cursor.fetchall()
        total_to_embed = len(rows_to_embed)

        console.print(f"[cyan]Found [bold]{total_to_embed:,}[/bold] reviews with text comments to embed.[/cyan]")
        if total_to_embed == 0:
            console.print("[green]All reviews already contain embeddings.[/green]")
            return

        batch_size = settings.EMBEDDING_BATCH_SIZE
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[magenta]Embedding reviews...", total=total_to_embed)

            for i in range(0, total_to_embed, batch_size):
                chunk = rows_to_embed[i : i + batch_size]
                review_ids = [r[0] for r in chunk]
                # Combine title and message for richer semantic context
                texts = [
                    f"{r[1] + ': ' if r[1] else ''}{r[2]}"
                    for r in chunk
                ]

                # Compute 768-d normalized embeddings
                embeddings = self.embedder.embed_documents(texts)

                # Batch update in DuckDB using temporary staging dataframe / values
                # Constructing batch update for DuckDB
                update_records = list(zip(embeddings, review_ids))
                conn.executemany(
                    "UPDATE order_reviews SET embedding = ? WHERE review_id = ?;",
                    update_records
                )
                progress.advance(task, len(chunk))

        console.print("[green][OK] Review embeddings successfully generated and persisted.[/green]")

    def build_indexes(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Compiles HNSW vector index and BM25 full-text search index."""
        console.print("[yellow]Building HNSW Vector Index (Plan B SIMD search)...[/yellow]")
        start_hnsw = time.perf_counter()
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reviews_embedding_hnsw
            ON order_reviews USING HNSW (embedding)
            WITH (metric = 'cosine', ef_construction = 128, M = 16);
        """)
        console.print(f"[green][OK] HNSW Index built in {time.perf_counter() - start_hnsw:.2f}s[/green]")

        console.print("[yellow]Building BM25 Full-Text Index (fts)...[/yellow]")
        start_fts = time.perf_counter()
        conn.execute("""
            PRAGMA create_fts_index(
                'order_reviews',
                'review_id',
                'review_comment_title',
                'review_comment_message',
                stemmer = 'english',
                stopwords = 'english',
                lower = 1,
                overwrite = 1
            );
        """)
        console.print(f"[green][OK] BM25 Index built in {time.perf_counter() - start_fts:.2f}s[/green]")

    def print_summary(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Display summary table of database size and row counts."""
        table = Table(title="DualQuery Unified DuckDB Telemetry", style="bold magenta")
        table.add_column("Table Name", style="cyan")
        table.add_column("Total Rows", justify="right", style="green")
        table.add_column("Rows with 768-d Vectors", justify="right", style="yellow")

        tables = ["customers", "orders", "order_items", "order_payments", "products", "order_reviews"]
        for t in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
            vector_count = "N/A"
            if t == "order_reviews":
                vector_count = f"{conn.execute('SELECT COUNT(*) FROM order_reviews WHERE embedding IS NOT NULL;').fetchone()[0]:,}"
            table.add_row(t, f"{count:,}", vector_count)

        console.print(table)
        db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
        console.print(f"[bold green]Total Database Size on Disk:[/bold green] [bold cyan]{db_size_mb:.2f} MB[/bold cyan]")

    def run(self, max_reviews: Optional[int] = None) -> None:
        """Execute full ingestion pipeline."""
        console.print(f"[bold magenta]Starting DualQuery Ingestion Pipeline on {self.db_path}...[/bold magenta]")
        conn = self.connect()
        try:
            self.initialize_schema(conn)
            self.load_relational_tables(conn)
            self.load_and_embed_reviews(conn, max_reviews=max_reviews)
            self.build_indexes(conn)
            self.print_summary(conn)
        finally:
            conn.close()


if __name__ == "__main__":
    loader = OlistLoader()
    loader.run()
