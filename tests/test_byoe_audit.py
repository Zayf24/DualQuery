"""
Unit and Integration tests for DualQuery BYOE Ingestion & Commercial Audit Engine.
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import duckdb

from dualquery.ingestion.byoe_adapter import BYOEAdapter, map_headers, TRANSACTION_ALIASES, FEEDBACK_ALIASES
from dualquery.engine.audit_runner import CommercialAuditRunner, DEFAULT_ROOT_CAUSE_PROTOTYPES


def test_fuzzy_header_mapping_disjoint():
    """Verify 3-tier cascade and 1-to-1 disjoint mapping on dirty headers."""
    tx_cols = ["Invoice_No", "Client_ID", "Product_Category", "Region", "Invoice_Amount", "Shipping_Fee", "Delivery_Status", "Invoice_Date", "Days_Late"]
    fb_cols = ["complaint_id", "invoice_no", "severity_rating", "customer_remarks", "complaint_date"]

    tx_map = map_headers(tx_cols, TRANSACTION_ALIASES)
    fb_map = map_headers(fb_cols, FEEDBACK_ALIASES)

    # Check key mappings
    assert tx_map["tx_id"] == "Invoice_No"
    assert tx_map["party_id"] == "Client_ID"
    assert tx_map["category"] == "Product_Category"
    assert tx_map["amount"] == "Invoice_Amount"
    assert tx_map["delay_days"] == "Days_Late"

    assert fb_map["note_id"] == "complaint_id"
    assert fb_map["tx_id"] == "invoice_no"
    assert fb_map["severity_score"] == "severity_rating"
    assert fb_map["raw_text"] == "customer_remarks"
    assert fb_map["note_timestamp"] == "complaint_date"

    # Verify non-ID field did not claim _id column
    assert fb_map["raw_text"] != "complaint_id"


def test_byoe_ingest_and_audit_pipeline(tmp_path: Path):
    """Verify end-to-end ingestion and audit execution on temporary dataset."""
    db_file = tmp_path / "test_byoe.duckdb"
    tx_file = tmp_path / "test_tx.csv"
    fb_file = tmp_path / "test_fb.csv"

    # Create synthetic test CSVs
    df_tx = pd.DataFrame([
        {"invoice_no": "INV-001", "client_id": "CLT-1", "product_category": "Furniture", "region": "West", "invoice_amount": 1000.0, "shipping_fee": 50.0, "delivery_status": "Returned", "invoice_date": "2025-01-01", "days_late": 5},
        {"invoice_no": "INV-002", "client_id": "CLT-2", "product_category": "Electronics", "region": "East", "invoice_amount": 2000.0, "shipping_fee": 30.0, "delivery_status": "Delivered", "invoice_date": "2025-01-02", "days_late": 0},
        {"invoice_no": "INV-003", "client_id": "CLT-3", "product_category": "Hardware", "region": "North", "invoice_amount": 1500.0, "shipping_fee": 40.0, "delivery_status": "Canceled", "invoice_date": "2025-01-03", "days_late": 10},
    ])
    df_tx.to_csv(tx_file, index=False)

    df_fb = pd.DataFrame([
        {"complaint_id": "N-1", "invoice_no": "INV-001", "severity_rating": 1, "customer_remarks": "The carton was crushed and shattered display. Call 555-019-2834.", "complaint_date": "2025-01-05"},
        {"complaint_id": "N-2", "invoice_no": "INV-003", "severity_rating": 2, "customer_remarks": "Late delivery stalled assembly line.", "complaint_date": "2025-01-06"},
    ])
    df_fb.to_csv(fb_file, index=False)

    # Ingest
    adapter = BYOEAdapter(db_path=db_file)
    report = adapter.ingest(tx_file, fb_file)

    assert report.transactions_ingested == 3
    assert report.feedback_notes_ingested == 2
    assert report.embedded_count == 2
    assert db_file.exists()

    # Run Audit
    runner = CommercialAuditRunner(db_path=db_file)
    audit_report = runner.run_audit()

    assert audit_report.total_transactions == 3
    assert audit_report.total_feedback_notes == 2
    assert audit_report.total_revenue_at_risk == 2500.0  # INV-001 (1000) + INV-003 (1500)
    assert audit_report.total_freight_loss == 90.0      # 50 + 40
    assert audit_report.total_financial_leakage == 2590.0

    # Verify PII scrubbing in sample evidence (phone number redacted)
    for c in audit_report.clusters:
        for evidence in c.sample_evidence:
            assert "555-019-2834" not in evidence
            if "crushed" in evidence:
                assert "[PHONE_REDACTED]" in evidence or "[REDACTED]" in evidence

