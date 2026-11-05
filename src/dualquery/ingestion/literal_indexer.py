"""
DualQuery Categorical Literal Indexer.
Extracts distinct enum literals (categories, states, payment types, statuses)
from DuckDB and compiles a lookup dictionary for Pre-SQL grounding.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List
import duckdb
from rich.console import Console
from rich.table import Table

from dualquery.config import settings

console = Console()


class LiteralIndexer:
    def __init__(self, db_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.db_path = db_path or settings.DUCKDB_PATH
        self.output_path = output_path or (settings.DATA_DIR / "literal_dictionary.json")

    def build_index(self) -> Dict[str, List[str]]:
        """Extract unique values from DuckDB categorical columns."""
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found at {self.db_path}. Please run olist_loader first.")

        console.print(f"[cyan]Extracting distinct literals from [bold]{self.db_path}[/bold]...[/cyan]")
        conn = duckdb.connect(str(self.db_path), read_only=True)
        literals: Dict[str, List[str]] = {}

        try:
            # 1. Customer States (e.g. SP, RJ, MG)
            states = [r[0] for r in conn.execute("""
                SELECT DISTINCT customer_state 
                FROM customers 
                WHERE customer_state IS NOT NULL 
                ORDER BY customer_state;
            """).fetchall()]
            literals["customer_state"] = states

            # 2. Product Categories (Portuguese and English)
            cat_pt = [r[0] for r in conn.execute("""
                SELECT DISTINCT product_category_name 
                FROM products 
                WHERE product_category_name IS NOT NULL 
                ORDER BY product_category_name;
            """).fetchall()]
            literals["product_category_name"] = cat_pt

            cat_en = [r[0] for r in conn.execute("""
                SELECT DISTINCT product_category_name_english 
                FROM products 
                WHERE product_category_name_english IS NOT NULL 
                ORDER BY product_category_name_english;
            """).fetchall()]
            literals["product_category_name_english"] = cat_en

            # 3. Payment Types (credit_card, boleto, voucher, debit_card)
            pay_types = [r[0] for r in conn.execute("""
                SELECT DISTINCT payment_type 
                FROM order_payments 
                WHERE payment_type IS NOT NULL 
                ORDER BY payment_type;
            """).fetchall()]
            literals["payment_type"] = pay_types

            # 4. Order Statuses (delivered, shipped, canceled, etc.)
            statuses = [r[0] for r in conn.execute("""
                SELECT DISTINCT order_status 
                FROM orders 
                WHERE order_status IS NOT NULL 
                ORDER BY order_status;
            """).fetchall()]
            literals["order_status"] = statuses

        finally:
            conn.close()

        # Save to disk
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(literals, f, indent=2, ensure_ascii=False)

        console.print(f"[green][OK] Literal dictionary saved to [bold]{self.output_path}[/bold][/green]")
        self.print_summary(literals)
        return literals

    def print_summary(self, literals: Dict[str, List[str]]) -> None:
        """Display summary of extracted enum literals."""
        table = Table(title="Extracted Pre-SQL Literal Domains", style="bold cyan")
        table.add_column("Domain / Column", style="magenta")
        table.add_column("Distinct Count", justify="right", style="green")
        table.add_column("Sample Literals", style="white")

        for domain, values in literals.items():
            sample = ", ".join(values[:5])
            if len(values) > 5:
                sample += f", ... (+{len(values) - 5} more)"
            table.add_row(domain, str(len(values)), sample)

        console.print(table)


if __name__ == "__main__":
    indexer = LiteralIndexer()
    indexer.build_index()
