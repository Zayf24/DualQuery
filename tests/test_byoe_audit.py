"""
Tests for BYOE CSV ingestion and the root-cause financial audit (fake embedder, no model download).
"""

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from dualquery.ingestion.byoe_adapter import BYOEAdapter, map_headers, TRANSACTION_ALIASES, FEEDBACK_ALIASES
from dualquery.engine.audit_runner import CommercialAuditRunner

PROTOTYPES = {"Damage": "broken", "Delay": "late"}


def write_csvs(folder: Path, tx_rows, fb_rows):
    tx, fb = folder / "tx.csv", folder / "fb.csv"
    pd.DataFrame(tx_rows).to_csv(tx, index=False)
    pd.DataFrame(fb_rows).to_csv(fb, index=False)
    return tx, fb


def tx(invoice, amount, freight, status, client="CLT-1"):
    return {"invoice_no": invoice, "client_id": client, "product_category": "Hardware", "region": "West",
            "invoice_amount": amount, "shipping_fee": freight, "delivery_status": status,
            "invoice_date": "2025-01-01", "days_late": 2}


def note(note_id, invoice, text, date="2025-01-05"):
    return {"complaint_id": note_id, "invoice_no": invoice, "severity_rating": 1,
            "customer_remarks": text, "complaint_date": date}


def audit(db, fake_embedder):
    return CommercialAuditRunner(db_path=db, embedder=fake_embedder, prototypes=PROTOTYPES).run_audit()


def test_fuzzy_header_mapping_disjoint():
    """3-tier cascade maps dirty headers 1-to-1 and never gives an _id column to a non-ID field."""
    tx_cols = ["Invoice_No", "Client_ID", "Product_Category", "Region", "Invoice_Amount", "Shipping_Fee", "Delivery_Status", "Invoice_Date", "Days_Late"]
    fb_cols = ["complaint_id", "invoice_no", "severity_rating", "customer_remarks", "complaint_date"]

    tx_map = map_headers(tx_cols, TRANSACTION_ALIASES)
    fb_map = map_headers(fb_cols, FEEDBACK_ALIASES)

    assert tx_map["tx_id"] == "Invoice_No"
    assert tx_map["party_id"] == "Client_ID"
    assert tx_map["category"] == "Product_Category"
    assert tx_map["amount"] == "Invoice_Amount"
    assert tx_map["delay_days"] == "Days_Late"
    assert fb_map == {"note_id": "complaint_id", "tx_id": "invoice_no", "severity_score": "severity_rating",
                      "raw_text": "customer_remarks", "note_timestamp": "complaint_date"}


def test_missing_required_columns_fail_loudly(tmp_path, fake_embedder):
    t, f = write_csvs(tmp_path, [{"foo": 1}], [{"bar": 2}])
    with pytest.raises(ValueError, match="transaction ID"):
        BYOEAdapter(db_path=tmp_path / "a.duckdb", embedder=fake_embedder).ingest(t, f)


def test_ingest_and_audit_pipeline(tmp_path, fake_embedder):
    db = tmp_path / "audit.duckdb"
    t, f = write_csvs(
        tmp_path,
        [tx("INV-001", 1000.0, 50.0, "Returned"), tx("INV-002", 2000.0, 30.0, "Delivered"), tx("INV-003", 1500.0, 40.0, "Canceled")],
        [note("N-1", "INV-001", "Box arrived broken. Call 555-019-2834."),
         note("N-2", "INV-003", "Late delivery stalled the line."),
         note("N-3", "INV-999", "orphan note for unknown invoice"),
         note("N-4", "INV-002", "   ")],
    )
    report = BYOEAdapter(db_path=db, embedder=fake_embedder).ingest(t, f)
    assert (report.transactions_ingested, report.feedback_notes_ingested, report.embedded_count) == (3, 2, 2)

    result = audit(db, fake_embedder)
    by_name = {c.cluster_name: c for c in result.clusters}
    assert result.total_transactions == 3 and result.total_feedback_notes == 2
    assert by_name["Damage"].gross_revenue_at_risk == 1000.0 and by_name["Delay"].gross_revenue_at_risk == 1500.0
    assert result.total_revenue_at_risk == 2500.0
    assert result.total_freight_loss == 90.0
    assert result.total_financial_leakage == 2590.0
    assert result.primary_leakage_driver == "Delay"
    evidence = " ".join(by_name["Damage"].sample_evidence)
    assert "555-019-2834" not in evidence and "[PHONE_REDACTED]" in evidence


def test_transaction_with_notes_in_two_clusters_is_counted_once(tmp_path, fake_embedder):
    db = tmp_path / "audit.duckdb"
    t, f = write_csvs(
        tmp_path,
        [tx("INV-001", 1000.0, 50.0, "Returned")],
        [note("N-1", "INV-001", "arrived broken", date="2025-01-05"),
         note("N-2", "INV-001", "and it was late", date="2025-01-07"),
         note("N-3", "INV-001", "really broken", date="2025-01-08")],
    )
    BYOEAdapter(db_path=db, embedder=fake_embedder).ingest(t, f)
    result = audit(db, fake_embedder)
    by_name = {c.cluster_name: c for c in result.clusters}

    assert by_name["Damage"].incident_count == 2 and by_name["Delay"].incident_count == 1
    # Money goes once, to the cluster of the earliest note (N-1, Damage).
    assert by_name["Damage"].gross_revenue_at_risk == 1000.0 and by_name["Damage"].total_freight_loss == 50.0
    assert by_name["Delay"].gross_revenue_at_risk == 0.0 and by_name["Delay"].impacted_orders == 0
    assert result.total_financial_leakage == 1050.0


def test_reingest_replaces_previous_dataset(tmp_path, fake_embedder):
    db = tmp_path / "audit.duckdb"
    adapter = BYOEAdapter(db_path=db, embedder=fake_embedder)
    adapter.ingest(*write_csvs(tmp_path, [tx("A-1", 10.0, 1.0, "Returned")], [note("N-1", "A-1", "broken")]))
    adapter.ingest(*write_csvs(tmp_path, [tx("A-1", 10.0, 1.0, "Returned")], [note("N-1", "A-1", "broken")]))  # same file twice
    adapter.ingest(*write_csvs(tmp_path, [tx("B-1", 99.0, 2.0, "Returned")], [note("N-9", "B-1", "late")]))

    with duckdb.connect(str(db), read_only=True) as conn:
        assert conn.execute("SELECT tx_id FROM canonical_transactions").fetchall() == [("B-1",)]
        assert conn.execute("SELECT note_id FROM canonical_feedback").fetchall() == [("N-9",)]
    assert audit(db, fake_embedder).total_revenue_at_risk == 99.0
