"""
DualQuery Empirical Optimizer Benchmark.
Evaluates cost-based physical planning decisions across the Golden Hybrid Query set.
Compares Plan A (SIMD Pushdown) vs Plan B (HNSW + BM25 RRF) on latency and evidence recall.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from dualquery.engine.executor import HybridExecutor
from dualquery.engine.physical_planner import HybridQueryRequest

console = Console()


def run_benchmark():
    benchmark_file = Path("benchmarks/golden_50_hybrid_queries.json")
    if not benchmark_file.exists():
        console.print(f"[bold red]Benchmark file not found: {benchmark_file}[/bold red]")
        return

    with open(benchmark_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    console.print(Panel.fit(
        f"[bold cyan]DualQuery Cost-Based Optimizer Benchmark[/bold cyan]\n"
        f"Evaluating [yellow]{len(queries)}[/yellow] compound relational-semantic queries\n"
        f"Comparing Plan A (SIMD Scan, N <= 5,000) vs Plan B (HNSW + BM25 RRF, N > 5,000)",
        border_style="cyan"
    ))

    executor = HybridExecutor()

    results: List[Dict[str, Any]] = []
    plan_a_latencies: List[float] = []
    plan_b_latencies: List[float] = []
    plan_match_count = 0

    table = Table(title="Optimizer Benchmark Execution Matrix", style="cyan", show_lines=True)
    table.add_column("Query ID", style="bold yellow", justify="center")
    table.add_column("Candidates (N)", justify="right", style="green")
    table.add_column("Expected", justify="center", style="white")
    table.add_column("Chosen", justify="center", style="cyan")
    table.add_column("Opt Latency (ms)", justify="right", style="magenta")
    table.add_column("Alt Latency (ms)", justify="right", style="dim")
    table.add_column("Citations (Opt vs Alt)", justify="center", style="blue")
    table.add_column("Status", justify="center")

    for q in queries:
        qid = q["id"]
        sql_filter = q["sql_filter"]
        semantic_query = q["semantic_query"]
        expected_plan = q["expected_plan"]
        aggregations = q.get("target_metrics", ["COUNT(*)"])

        req = HybridQueryRequest(
            sql_filter=sql_filter,
            semantic_query=semantic_query,
            aggregations=aggregations,
            limit=5
        )

        # 1. Execute with Adaptive Optimizer
        res_opt = executor.execute(req)
        opt_chosen = res_opt.plan_type
        opt_lat = res_opt.total_latency_ms
        n_cand = res_opt.estimated_candidates

        # 2. Execute with Forced Alternative Plan
        alt_plan = "Plan B" if opt_chosen == "Plan A" else "Plan A"
        try:
            res_alt = executor.execute(req, force_plan=alt_plan)
            alt_lat = res_alt.total_latency_ms
            alt_cits_count = len(res_alt.citations)
        except Exception as e:
            alt_lat = -1.0
            alt_cits_count = 0

        opt_cits_count = len(res_opt.citations)

        # Track latencies for chosen plans
        if opt_chosen == "Plan A":
            plan_a_latencies.append(opt_lat)
        else:
            plan_b_latencies.append(opt_lat)

        matched = (opt_chosen == expected_plan)
        if matched:
            plan_match_count += 1

        status = "[green][OK][/green]" if matched else "[red][FAIL][/red]"

        table.add_row(
            qid,
            f"{n_cand:,}",
            expected_plan,
            f"[bold]{opt_chosen}[/bold]",
            f"{opt_lat:.1f}",
            f"{alt_lat:.1f}" if alt_lat > 0 else "ERR",
            f"{opt_cits_count} vs {alt_cits_count}",
            status
        )

        results.append({
            "id": qid,
            "description": q["description"],
            "candidates": n_cand,
            "expected_plan": expected_plan,
            "chosen_plan": opt_chosen,
            "plan_matched": matched,
            "opt_latency_ms": opt_lat,
            "alt_latency_ms": alt_lat,
            "opt_citations_count": opt_cits_count,
            "alt_citations_count": alt_cits_count,
            "aggregates": {k: str(v) for k, v in res_opt.aggregates.items()}
        })

    console.print(table)

    # Summary Statistics
    accuracy = (plan_match_count / len(queries)) * 100
    p50_a = float(np.percentile(plan_a_latencies, 50)) if plan_a_latencies else 0.0
    p95_a = float(np.percentile(plan_a_latencies, 95)) if plan_a_latencies else 0.0
    p50_b = float(np.percentile(plan_b_latencies, 50)) if plan_b_latencies else 0.0
    p95_b = float(np.percentile(plan_b_latencies, 95)) if plan_b_latencies else 0.0

    summary_table = Table(title="Optimizer Benchmark Performance Summary", style="green")
    summary_table.add_column("Metric", style="bold yellow")
    summary_table.add_column("Plan A (High Selectivity, N <= 5,000)", style="cyan", justify="right")
    summary_table.add_column("Plan B (Low Selectivity, N > 5,000)", style="magenta", justify="right")

    summary_table.add_row("Decision Accuracy", f"{accuracy:.1f}%", f"{accuracy:.1f}%")
    summary_table.add_row("P50 Turn Latency (ms)", f"{p50_a:.2f}", f"{p50_b:.2f}")
    summary_table.add_row("P95 Turn Latency (ms)", f"{p95_a:.2f}", f"{p95_b:.2f}")
    summary_table.add_row("Queries Evaluated", str(len(plan_a_latencies)), str(len(plan_b_latencies)))

    console.print(summary_table)

    # Save artifact
    output_path = Path("benchmarks/optimizer_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "accuracy_percent": accuracy,
            "plan_a": {"p50_ms": p50_a, "p95_ms": p95_a, "sample_count": len(plan_a_latencies)},
            "plan_b": {"p50_ms": p50_b, "p95_ms": p95_b, "sample_count": len(plan_b_latencies)},
            "queries": results
        }, f, indent=2)

    console.print(f"[bold green]Saved benchmark results to {output_path}[/bold green]")


if __name__ == "__main__":
    run_benchmark()
