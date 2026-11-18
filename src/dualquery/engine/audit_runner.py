"""
DualQuery Commercial BYOE Audit Engine & Telemetry UI.
Performs semantic root-cause clustering over qualitative feedback notes and
correlates clusters with relational transactions to compute financial leakage.
"""

import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd
import duckdb
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

from dualquery.config import settings
from dualquery.ingestion.embeddings import get_embedding_provider, EmbeddingProvider
from dualquery.security.pii_vault import PIIVault

logger = logging.getLogger("dualquery.audit")

# Canonical Commercial Root-Cause Cluster Prototypes
DEFAULT_ROOT_CAUSE_PROTOTYPES: Dict[str, str] = {
    "Packaging & Transit Damage": (
        "crushed carton shattered display bent broken chassis packaging liquid leaked "
        "dropped by courier chipped scratched insufficient bubble wrap transit damage"
    ),
    "Missing Items & Accessories": (
        "missing power brick charging cable missing screws bolts mounting bracket "
        "absent units delivery challan remote control missing incomplete package short shipment"
    ),
    "Logistics Delays & Stalls": (
        "shipment arrived late past committed deadline assembly line stalled delayed delivery "
        "courier tracking no movement overdue delivery transit stall"
    ),
    "Specification & Variant Mismatches": (
        "ordered 220V received 110V wrong model wrong color wrong dimension black XL grey medium "
        "catalog model mismatch obsolete variant incorrect item delivered"
    ),
    "Component & Out-of-Box Defects": (
        "defective power supply circuit breaker trip textile seams came apart hydraulic pump "
        "overheats counterfeit motor seized warranty replacement manufacturing quality defect"
    ),
}


class ClusterAuditResult(BaseModel):
    """Detailed financial and operational impact of a root-cause cluster."""
    cluster_name: str
    incident_count: int
    impacted_orders: int
    impacted_parties: int
    gross_revenue_at_risk: float
    total_freight_loss: float
    avg_severity: float
    avg_delay_days: float
    top_category: str
    top_region: str
    sample_evidence: List[str] = Field(default_factory=list)


class CommercialAuditReport(BaseModel):
    """Executive enterprise audit report quantifying root-cause leakage."""
    database_path: str
    audit_timestamp: str
    total_transactions: int
    total_feedback_notes: int
    total_revenue_at_risk: float
    total_freight_loss: float
    total_financial_leakage: float
    primary_leakage_driver: str
    clusters: List[ClusterAuditResult]


class CommercialAuditRunner:
    """
    Automated Root-Cause Correlation and Financial Leakage Audit Engine.
    Correlates unstructured customer feedback notes with structured transactional data.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        embedder: Optional[EmbeddingProvider] = None,
        prototypes: Optional[Dict[str, str]] = None
    ):
        self.db_path = Path(db_path or (settings.DATA_DIR / "byoe_audit.duckdb"))
        self.embedder = embedder or get_embedding_provider("auto")
        self.prototypes = prototypes or DEFAULT_ROOT_CAUSE_PROTOTYPES
        self.pii_vault = PIIVault()

    def run_audit(self) -> CommercialAuditReport:
        """
        Executes semantic clustering and relational financial aggregation.
        """
        if not self.db_path.exists():
            raise FileNotFoundError(f"DuckDB database not found at {self.db_path}. Ingest data first.")

        # 1. Embed Cluster Prototypes
        proto_names = list(self.prototypes.keys())
        proto_texts = list(self.prototypes.values())
        logger.info(f"Generating embeddings for {len(proto_names)} root-cause prototypes...")
        raw_proto_vecs = np.array([self.embedder.embed_query(t) for t in proto_texts])
        proto_vecs = raw_proto_vecs / np.linalg.norm(raw_proto_vecs, axis=1, keepdims=True)

        # 2. Fetch Feedback Notes & Vectors from DuckDB
        conn = duckdb.connect(str(self.db_path))
        try:
            df_notes = conn.execute("""
                SELECT note_id, tx_id, severity_score, raw_text, note_timestamp, embedding
                FROM canonical_feedback
            """).fetchdf()

            if df_notes.empty:
                raise ValueError("canonical_feedback table is empty.")

            # 3. Vectorized Semantic Assignment
            raw_embs = np.array(df_notes["embedding"].tolist())
            norm_embs = raw_embs / np.linalg.norm(raw_embs, axis=1, keepdims=True)

            sim_matrix = norm_embs @ proto_vecs.T  # (N, K)
            best_cluster_indices = np.argmax(sim_matrix, axis=1)
            best_scores = np.max(sim_matrix, axis=1)

            df_notes["cluster_name"] = [proto_names[i] for i in best_cluster_indices]
            df_notes["similarity_score"] = best_scores

            # Register classified feedback into DuckDB view
            conn.register("classified_feedback_view", df_notes)

            # 4. Relational Aggregation Across ERP Invoices & Returns.
            # A transaction can have several notes, possibly in different clusters. Its money is
            # attributed once, to the cluster of its earliest note, so cluster totals add up exactly.
            agg_query = """
                WITH tx_cluster AS (
                    SELECT tx_id, cluster_name FROM (
                        SELECT tx_id, cluster_name, ROW_NUMBER() OVER (
                            PARTITION BY tx_id ORDER BY note_timestamp NULLS LAST, note_id
                        ) AS rn
                        FROM classified_feedback_view
                    ) WHERE rn = 1
                ),
                incidents AS (
                    SELECT cluster_name,
                           COUNT(DISTINCT note_id) AS incident_count,
                           ROUND(AVG(severity_score), 2) AS avg_severity
                    FROM classified_feedback_view
                    GROUP BY cluster_name
                ),
                money AS (
                    SELECT tc.cluster_name,
                           COUNT(DISTINCT ct.tx_id) AS impacted_orders,
                           COUNT(DISTINCT ct.party_id) AS impacted_parties,
                           SUM(CASE WHEN ct.status IN ('Returned', 'Canceled') THEN ct.amount ELSE 0 END) AS gross_revenue_at_risk,
                           SUM(ct.freight_or_tax) AS total_freight_loss,
                           ROUND(AVG(ct.delay_days), 1) AS avg_delay_days,
                           MODE(ct.category) AS top_category,
                           MODE(ct.region) AS top_region
                    FROM tx_cluster tc
                    JOIN canonical_transactions ct ON tc.tx_id = ct.tx_id
                    GROUP BY tc.cluster_name
                )
                SELECT
                    i.cluster_name,
                    i.incident_count,
                    COALESCE(m.impacted_orders, 0) AS impacted_orders,
                    COALESCE(m.impacted_parties, 0) AS impacted_parties,
                    COALESCE(m.gross_revenue_at_risk, 0.0) AS gross_revenue_at_risk,
                    COALESCE(m.total_freight_loss, 0.0) AS total_freight_loss,
                    COALESCE(i.avg_severity, 0.0) AS avg_severity,
                    COALESCE(m.avg_delay_days, 0.0) AS avg_delay_days,
                    COALESCE(m.top_category, 'n/a') AS top_category,
                    COALESCE(m.top_region, 'n/a') AS top_region
                FROM incidents i
                LEFT JOIN money m ON i.cluster_name = m.cluster_name
                ORDER BY gross_revenue_at_risk DESC, i.cluster_name;
            """
            df_cluster_metrics = conn.execute(agg_query).fetchdf()

            # Global Totals
            total_tx = conn.execute("SELECT COUNT(*) FROM canonical_transactions").fetchone()[0]
            total_notes = len(df_notes)

            # Extract Scrubbed Verbatim Evidence per Cluster
            cluster_results = []
            for _, row in df_cluster_metrics.iterrows():
                c_name = row["cluster_name"]
                # Get top 2 sample remarks for this cluster
                samples = df_notes[df_notes["cluster_name"] == c_name]["raw_text"].head(2).tolist()
                scrubbed_samples = [self.pii_vault.scrub_text(s).sanitized_text for s in samples]

                cluster_results.append(
                    ClusterAuditResult(
                        cluster_name=c_name,
                        incident_count=int(row["incident_count"]),
                        impacted_orders=int(row["impacted_orders"]),
                        impacted_parties=int(row["impacted_parties"]),
                        gross_revenue_at_risk=round(float(row["gross_revenue_at_risk"]), 2),
                        total_freight_loss=round(float(row["total_freight_loss"]), 2),
                        avg_severity=float(row["avg_severity"]),
                        avg_delay_days=float(row["avg_delay_days"]),
                        top_category=str(row["top_category"]),
                        top_region=str(row["top_region"]),
                        sample_evidence=scrubbed_samples,
                    )
                )

            total_risk = sum(c.gross_revenue_at_risk for c in cluster_results)
            total_freight = sum(c.total_freight_loss for c in cluster_results)
            total_leakage = total_risk + total_freight
            primary_driver = cluster_results[0].cluster_name if cluster_results else "None"

            return CommercialAuditReport(
                database_path=str(self.db_path),
                audit_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                total_transactions=total_tx,
                total_feedback_notes=total_notes,
                total_revenue_at_risk=round(total_risk, 2),
                total_freight_loss=round(total_freight, 2),
                total_financial_leakage=round(total_leakage, 2),
                primary_leakage_driver=primary_driver,
                clusters=cluster_results,
            )
        finally:
            conn.close()

    def render_dashboard(self, report: CommercialAuditReport, console: Optional[Console] = None) -> None:
        """
        Renders a Bloomberg-grade executive telemetry dashboard in the terminal.
        """
        console = console or Console()

        # Header Title
        console.print()
        title_text = Text("DUALQUERY COMMERCIAL BYOE FINANCIAL AUDIT", style="bold white on blue")
        console.print(Panel(title_text, expand=False, border_style="blue"))

        # Executive Metrics Cards
        claim_rate = (report.total_feedback_notes / max(1, report.total_transactions)) * 100
        metric_panel_content = (
            f"[bold cyan]Total Ingested Transactions:[/bold cyan] {report.total_transactions:,}\n"
            f"[bold cyan]Audited Complaint Notes:[/bold cyan] {report.total_feedback_notes:,} ({claim_rate:.1f}% Incident Rate)\n"
            f"[bold red]Total Gross Revenue at Risk:[/bold red] ${report.total_revenue_at_risk:,.2f}\n"
            f"[bold yellow]Total Logistics / Freight Loss:[/bold yellow] ${report.total_freight_loss:,.2f}\n"
            f"[bold green]Total Financial Leakage:[/bold green] ${report.total_financial_leakage:,.2f}\n"
            f"[bold magenta]Primary Leakage Driver:[/bold magenta] {report.primary_leakage_driver}"
        )
        console.print(Panel(metric_panel_content, title="[bold]Executive Summary[/bold]", border_style="green"))

        # Root Cause Financial Matrix Table
        table = Table(
            title="[bold]Root-Cause Semantic Clusters & Financial Attribution Matrix[/bold]",
            header_style="bold cyan",
            border_style="dim",
            show_lines=True,
            expand=True
        )

        table.add_column("Root Cause Cluster", style="bold white", ratio=4)
        table.add_column("Incidents", justify="right", ratio=1)
        table.add_column("Orders", justify="right", ratio=1)
        table.add_column("Clients", justify="right", ratio=1)
        table.add_column("Revenue at Risk", justify="right", style="bold red", ratio=2)
        table.add_column("Freight Loss", justify="right", style="yellow", ratio=2)
        table.add_column("Avg Delay", justify="right", ratio=1)
        table.add_column("Top Category", style="magenta", ratio=2)
        table.add_column("Top Region", style="cyan", ratio=2)

        for c in report.clusters:
            table.add_row(
                c.cluster_name,
                str(c.incident_count),
                str(c.impacted_orders),
                str(c.impacted_parties),
                f"${c.gross_revenue_at_risk:,.2f}",
                f"${c.total_freight_loss:,.2f}",
                f"{c.avg_delay_days:.1f}d",
                c.top_category,
                c.top_region,
            )

        console.print(table)

        # Drill-down Evidence Panel
        evidence_panel_lines = []
        for c in report.clusters:
            evidence_panel_lines.append(f"[bold underline cyan]{c.cluster_name}[/bold underline cyan] (${c.gross_revenue_at_risk:,.2f} at risk):")
            for idx, quote in enumerate(c.sample_evidence, 1):
                evidence_panel_lines.append(f'  [dim]#{idx}:[/dim] "{quote}"')
            evidence_panel_lines.append("")

        console.print(Panel("\n".join(evidence_panel_lines).strip(), title="[bold]Scrubbed Qualitative Evidence (Zero-Trust Verified)[/bold]", border_style="cyan"))
        console.print(f"[dim]Audit generated at {report.audit_timestamp} | Storage: {report.database_path}[/dim]\n")

