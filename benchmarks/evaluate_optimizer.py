"""
DualQuery Optimizer Latency Benchmark.

Runs every golden query twice: with the plan the optimizer picks, and with the other plan forced.
Reports per-plan latency (end-to-end engine turn and retrieval SQL alone) and how often the
optimizer's choice was also the faster plan. Each query is run once untimed first to warm caches.

Run: uv run python benchmarks/evaluate_optimizer.py
"""

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from dualquery.engine.executor import HybridExecutor
from dualquery.engine.physical_planner import HybridQueryRequest

console = Console()


def pct(values, p):
    return round(float(np.percentile(values, p)), 1) if values else 0.0


def run_benchmark():
    queries = json.loads(Path("benchmarks/golden_hybrid_queries.json").read_text(encoding="utf-8"))
    executor = HybridExecutor()
    results = []

    table = Table(title="Optimizer: chosen plan vs forced alternative", show_lines=False)
    for col in ("Query", "N_cand", "Chosen", "Chosen ms", "Other ms", "Faster?", "Citations"):
        table.add_column(col, justify="right")

    for q in queries:
        req = HybridQueryRequest(
            sql_filter=q["sql_filter"],
            semantic_query=q["semantic_query"],
            aggregations=q.get("target_metrics", ["COUNT(*)"]),
            limit=5,
        )
        executor.execute(req)  # warm-up (model, OS page cache)
        chosen = executor.execute(req)
        other_plan = "Plan B" if chosen.plan_type == "Plan A" else "Plan A"
        other = executor.execute(req, force_plan=other_plan)
        faster = chosen.engine_latency_ms <= other.engine_latency_ms

        results.append({
            "id": q["id"],
            "candidates": chosen.estimated_candidates,
            "chosen_plan": chosen.plan_type,
            "chosen_total_ms": chosen.total_latency_ms,
            "chosen_retrieval_ms": chosen.engine_latency_ms,
            "other_total_ms": other.total_latency_ms,
            "other_retrieval_ms": other.engine_latency_ms,
            "chose_faster_retrieval": faster,
            "citations": len(chosen.citations),
            "matched_orders": chosen.matched_orders,
            "aggregates": {k: str(v) for k, v in chosen.aggregates.items()},
        })
        table.add_row(q["id"], f"{chosen.estimated_candidates:,}", chosen.plan_type,
                      f"{chosen.engine_latency_ms:.0f}", f"{other.engine_latency_ms:.0f}",
                      "yes" if faster else "NO", str(len(chosen.citations)))

    console.print(table)

    summary = {}
    for plan in ("Plan A", "Plan B"):
        rows = [r for r in results if r["chosen_plan"] == plan]
        summary[plan] = {
            "queries": len(rows),
            "total_p50_ms": pct([r["chosen_total_ms"] for r in rows], 50),
            "total_p95_ms": pct([r["chosen_total_ms"] for r in rows], 95),
            "retrieval_p50_ms": pct([r["chosen_retrieval_ms"] for r in rows], 50),
            "retrieval_p95_ms": pct([r["chosen_retrieval_ms"] for r in rows], 95),
            "other_plan_retrieval_p50_ms": pct([r["other_retrieval_ms"] for r in rows], 50),
            "chose_faster_retrieval": sum(r["chose_faster_retrieval"] for r in rows),
            "full_citations": sum(r["citations"] == 5 for r in rows),
        }
        console.print(f"{plan}: {summary[plan]}")

    out = Path("benchmarks/optimizer_benchmark_results.json")
    out.write_text(json.dumps({"summary": summary, "queries": results}, indent=2), encoding="utf-8")
    console.print(f"[bold green]Saved {out}[/bold green]")


if __name__ == "__main__":
    run_benchmark()
