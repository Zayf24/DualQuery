"""
DualQuery Commercial Bring-Your-Own-Export (BYOE) Universal Ingestion Adapter.
Maps arbitrary 2-table business CSV exports (invoices/orders + return/complaint notes)
into unified DuckDB tables with 768-d note embeddings.
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import duckdb
from pydantic import BaseModel, Field

from dualquery.config import settings
from dualquery.ingestion.embeddings import get_embedding_provider, EmbeddingProvider

from rapidfuzz import fuzz

logger = logging.getLogger("dualquery.byoe")


# Intelligent fuzzy header aliases
TRANSACTION_ALIASES = {
    "tx_id": ["tx_id", "invoice_no", "invoice_id", "order_id", "bill_no", "order_number", "id"],
    "party_id": ["party_id", "customer_id", "vendor_id", "client_id", "account_id", "buyer_id", "account", "buyer", "customer"],
    "category": ["category", "product_category", "item_group", "type", "department", "segment"],
    "region": ["region", "state", "city", "zone", "territory", "location"],
    "amount": ["amount", "total", "net_amount", "invoice_amount", "invoice_value", "total_amount", "value", "price"],
    "freight_or_tax": ["freight_or_tax", "shipping_fee", "freight", "tax", "shipping", "shipping_cost"],
    "status": ["status", "delivery_status", "order_status", "invoice_status"],
    "tx_timestamp": ["tx_timestamp", "invoice_date", "order_date", "date", "timestamp"],
    "delay_days": ["delay_days", "days_late", "delay", "overdue_days"],
}

FEEDBACK_ALIASES = {
    "note_id": ["note_id", "feedback_id", "ticket_id", "complaint_id", "return_id", "note_no", "id"],
    "tx_id": ["tx_id", "invoice_no", "invoice_id", "order_id", "bill_no", "order_number"],
    "severity_score": ["severity_score", "score", "rating", "severity", "stars", "priority"],
    "raw_text": ["raw_text", "remarks", "customer_remarks", "comment", "feedback_text", "review_comment", "notes", "description", "complaint_text", "review", "reason", "message", "text"],
    "note_timestamp": ["note_timestamp", "complaint_date", "created_at", "date", "timestamp"],
}


class IngestionReport(BaseModel):
    """Telemetry report of a completed BYOE ingestion."""
    database_path: str
    transactions_ingested: int
    feedback_notes_ingested: int
    embedded_count: int
    duration_seconds: float
    column_mappings: Dict[str, Dict[str, str]]


def map_headers(columns: List[str], alias_dict: Dict[str, List[str]], threshold: float = 75.0) -> Dict[str, str]:
    """
    Map CSV column headers to canonical schema names using a 3-tier cascade:
    1. Exact token match (e.g., 'invoice_no' == 'invoice_no')
    2. Substring / Token containment (e.g., 'remarks' in 'customer_remarks')
    3. RapidFuzz token_set_ratio similarity (> threshold)

    Enforces 1-to-1 disjoint column mapping and prevents non-ID fields from claiming '_id' columns.
    """
    mapping = {}
    used_cols = set()
    col_map = {col.strip().lower(): col for col in columns}

    for canonical, aliases in alias_dict.items():
        best_match = None
        best_score = 0.0

        # Available candidate columns (not already mapped)
        avail_cols = {c_low: orig for c_low, orig in col_map.items() if orig not in used_cols}

        # Check if canonical field represents an ID
        is_id_field = canonical.endswith("_id") or canonical in ("tx_id", "party_id", "note_id")

        # Tier 1: Exact match
        for alias in aliases:
            if alias in avail_cols:
                best_match = avail_cols[alias]
                best_score = 100.0
                break

        # Tier 2: Substring containment (prefer longer aliases)
        if not best_match:
            sorted_aliases = sorted(aliases, key=len, reverse=True)
            for alias in sorted_aliases:
                for col_lower, orig_col in avail_cols.items():
                    if not is_id_field and (col_lower.endswith("_id") or col_lower == "id"):
                        continue
                    if alias in col_lower or col_lower in alias:
                        best_match = orig_col
                        best_score = 90.0
                        break
                if best_match:
                    break

        # Tier 3: RapidFuzz token similarity
        if not best_match:
            for col_lower, orig_col in avail_cols.items():
                if not is_id_field and (col_lower.endswith("_id") or col_lower == "id"):
                    continue
                for alias in aliases:
                    score = fuzz.token_set_ratio(alias, col_lower)
                    if score >= threshold and score > best_score:
                        best_score = score
                        best_match = orig_col

        if best_match:
            mapping[canonical] = best_match
            used_cols.add(best_match)

    return mapping


class BYOEAdapter:
    """
    Universal 2-Table Enterprise Ingestion Adapter.
    Ingests transactions and qualitative notes into indexed DuckDB storage.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedder: Optional[EmbeddingProvider] = None
    ):
        self.db_path = Path(db_path or (settings.DATA_DIR / "byoe_audit.duckdb"))
        self.embedder = embedder or get_embedding_provider("auto")

    def init_database(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Create the canonical transaction and feedback tables."""
        conn.execute("""
            CREATE TABLE canonical_transactions (
                tx_id VARCHAR PRIMARY KEY,
                party_id VARCHAR,
                category VARCHAR,
                region VARCHAR,
                amount DECIMAL(12, 2),
                freight_or_tax DECIMAL(10, 2),
                status VARCHAR,
                tx_timestamp TIMESTAMP,
                delay_days INTEGER
            );

            CREATE TABLE canonical_feedback (
                note_id VARCHAR PRIMARY KEY,
                tx_id VARCHAR REFERENCES canonical_transactions(tx_id),
                severity_score TINYINT,
                raw_text VARCHAR,
                note_timestamp TIMESTAMP,
                embedding FLOAT[768]
            );
        """)

    def ingest(
        self,
        transactions_csv: Path,
        feedback_csv: Path,
        batch_size: int = 128
    ) -> IngestionReport:
        """
        Load, map, embed, and index arbitrary commercial CSV files.
        """
        start_time = time.perf_counter()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading transactions from {transactions_csv}")
        df_tx = pd.read_csv(transactions_csv)

        logger.info(f"Loading feedback notes from {feedback_csv}")
        df_fb = pd.read_csv(feedback_csv)

        # 1. Deduce Column Mappings
        tx_mapping = map_headers(list(df_tx.columns), TRANSACTION_ALIASES)
        fb_mapping = map_headers(list(df_fb.columns), FEEDBACK_ALIASES)

        logger.info(f"Deduced Transaction mappings: {tx_mapping}")
        logger.info(f"Deduced Feedback mappings: {fb_mapping}")

        # Check required primary/foreign keys
        if "tx_id" not in tx_mapping:
            raise ValueError(f"Could not identify a transaction ID column in {df_tx.columns}")
        if "note_id" not in fb_mapping or "raw_text" not in fb_mapping:
            raise ValueError(f"Could not identify note_id or raw_text column in {df_fb.columns}")

        # 2. Transform Transactions DataFrame to Canonical Schema
        canonical_tx = pd.DataFrame()
        canonical_tx["tx_id"] = df_tx[tx_mapping["tx_id"]].astype(str)
        canonical_tx["party_id"] = df_tx[tx_mapping["party_id"]].astype(str) if "party_id" in tx_mapping else "UNKNOWN"
        canonical_tx["category"] = df_tx[tx_mapping["category"]].astype(str) if "category" in tx_mapping else "general"
        canonical_tx["region"] = df_tx[tx_mapping["region"]].astype(str) if "region" in tx_mapping else "NATIONAL"
        canonical_tx["amount"] = pd.to_numeric(df_tx[tx_mapping["amount"]], errors="coerce").fillna(0.0) if "amount" in tx_mapping else 0.0
        canonical_tx["freight_or_tax"] = pd.to_numeric(df_tx[tx_mapping["freight_or_tax"]], errors="coerce").fillna(0.0) if "freight_or_tax" in tx_mapping else 0.0
        canonical_tx["status"] = df_tx[tx_mapping["status"]].astype(str) if "status" in tx_mapping else "completed"
        canonical_tx["tx_timestamp"] = pd.to_datetime(df_tx[tx_mapping["tx_timestamp"]], errors="coerce") if "tx_timestamp" in tx_mapping else pd.Timestamp.now()
        canonical_tx["delay_days"] = pd.to_numeric(df_tx[tx_mapping["delay_days"]], errors="coerce").fillna(0).astype(int) if "delay_days" in tx_mapping else 0

        # Drop duplicate primary keys
        canonical_tx = canonical_tx.drop_duplicates(subset=["tx_id"])

        # 3. Transform Feedback DataFrame to Canonical Schema
        canonical_fb = pd.DataFrame()
        canonical_fb["note_id"] = df_fb[fb_mapping["note_id"]].astype(str)
        canonical_fb["tx_id"] = df_fb[fb_mapping["tx_id"]].astype(str) if "tx_id" in fb_mapping else None
        canonical_fb["severity_score"] = pd.to_numeric(df_fb[fb_mapping["severity_score"]], errors="coerce").fillna(3).astype(int) if "severity_score" in fb_mapping else 3
        canonical_fb["raw_text"] = df_fb[fb_mapping["raw_text"]].fillna("").astype(str)
        canonical_fb["note_timestamp"] = pd.to_datetime(df_fb[fb_mapping["note_timestamp"]], errors="coerce") if "note_timestamp" in fb_mapping else pd.Timestamp.now()

        # Filter out empty comments and orphaned foreign keys
        valid_tx_ids = set(canonical_tx["tx_id"])
        canonical_fb = canonical_fb[canonical_fb["tx_id"].isin(valid_tx_ids)]
        canonical_fb = canonical_fb[canonical_fb["raw_text"].str.strip() != ""]
        canonical_fb = canonical_fb.drop_duplicates(subset=["note_id"])

        # 4. Generate Embeddings for Qualitative Feedback Notes
        logger.info(f"Generating 768-d embeddings for {len(canonical_fb)} feedback notes...")
        texts = canonical_fb["raw_text"].tolist()
        embeddings = self.embedder.embed_documents(texts, batch_size=batch_size)
        canonical_fb["embedding"] = embeddings

        # 5. Write a fresh DuckDB file: each ingest replaces the previous dataset instead of merging into it.
        for stale in (self.db_path, self.db_path.with_name(self.db_path.name + ".wal")):
            stale.unlink(missing_ok=True)
        conn = duckdb.connect(str(self.db_path))
        try:
            self.init_database(conn)

            logger.info(f"Writing {len(canonical_tx)} rows into canonical_transactions...")
            conn.register("df_tx_view", canonical_tx)
            conn.execute("INSERT INTO canonical_transactions SELECT * FROM df_tx_view")

            logger.info(f"Writing {len(canonical_fb)} rows into canonical_feedback...")
            conn.register("df_fb_view", canonical_fb)
            conn.execute("INSERT INTO canonical_feedback SELECT * FROM df_fb_view")
            conn.checkpoint()
        finally:
            conn.close()

        duration = time.perf_counter() - start_time
        return IngestionReport(
            database_path=str(self.db_path),
            transactions_ingested=len(canonical_tx),
            feedback_notes_ingested=len(canonical_fb),
            embedded_count=len(embeddings),
            duration_seconds=round(duration, 2),
            column_mappings={"transactions": tx_mapping, "feedback": fb_mapping}
        )

