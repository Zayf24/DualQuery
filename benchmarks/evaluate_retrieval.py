"""
DualQuery Filtered-ANN Retrieval Benchmark.

For every golden query, compares the vector leg of Plan B (HNSW top-k, then the SQL filter)
against exact filtered search (Plan A: SQL filter, then exact cosine scan = ground truth):

- recall@10: share of the exact filtered top-10 reviews that the ANN leg also returns
- fill@10:   how many of the 10 result slots the ANN leg can fill after filtering
- ndcg@10:   rank-aware agreement with the exact ordering

Two pool policies are compared: the original fixed pool (150) and the selectivity-sized pool
used by PhysicalPlanner.pool_size. Also asserts that the ANN query really uses the HNSW index.

Run: uv run python benchmarks/evaluate_retrieval.py
"""

import json
import math
import time
from pathlib import Path

import numpy as np

from dualquery.config import settings
from dualquery.engine.executor import HybridExecutor
from dualquery.engine.physical_planner import PhysicalPlanner
from dualquery.engine.sql_normalizer import get_required_joins
from dualquery.security.ast_guard import ANALYST_POLICY

K = 10
FIXED_POOL = 150
LIMIT = settings.PLAN_A_CARDINALITY_LIMIT


def ann_top_k(conn, vec, joins, sql_filter, pool):
    sql = f"""
        WITH hits AS (
            SELECT review_id, order_id, array_cosine_distance(embedding, ?::FLOAT[768]) AS dist
            FROM order_reviews ORDER BY dist LIMIT {pool}
        )
        SELECT DISTINCT hits.review_id, hits.dist FROM hits
        JOIN orders ON hits.order_id = orders.order_id {joins}
        WHERE {sql_filter} AND hits.dist IS NOT NULL
        ORDER BY hits.dist, hits.review_id LIMIT {K}
    """
    return [r[0] for r in conn.execute(sql, [vec]).fetchall()], sql


def exact_top_k(conn, vec, joins, sql_filter):
    sql = f"""
        SELECT r.review_id FROM order_reviews r
        WHERE r.embedding IS NOT NULL AND r.order_id IN (
            SELECT orders.order_id FROM orders {joins} WHERE {sql_filter})
        ORDER BY array_cosine_distance(r.embedding, ?::FLOAT[768]), r.review_id LIMIT {K}
    """
    return [r[0] for r in conn.execute(sql, [vec]).fetchall()]


def ndcg(pred, truth):
    rel = {rid: 1.0 for rid in truth}
    dcg = sum(rel.get(rid, 0.0) / math.log2(i + 2) for i, rid in enumerate(pred))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(len(truth)))
    return dcg / idcg if idcg else 1.0


def run():
    queries = json.loads(Path("benchmarks/golden_hybrid_queries.json").read_text(encoding="utf-8"))
    executor = HybridExecutor()
    conn = executor.connect()
    total_orders = conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
    rows = []
    try:
        for q in queries:
            safe_filter = executor.validate_filter(q["sql_filter"], ANALYST_POLICY)
            joins = get_required_joins(safe_filter)
            vec = executor.embedder.embed_query(q["semantic_query"])
            est = executor.planner.probe.estimate(conn, safe_filter)
            truth = exact_top_k(conn, vec, joins, safe_filter)
            if not truth:
                continue
            row = {"id": q["id"], "candidates": est.estimated_candidates,
                   "selectivity": est.estimated_candidates / total_orders}
            for name, pool in (("fixed_150", FIXED_POOL), ("adaptive", PhysicalPlanner.pool_size(est))):
                start = time.perf_counter()
                pred, sql = ann_top_k(conn, vec, joins, safe_filter, pool)
                ms = (time.perf_counter() - start) * 1000
                plan_text = conn.execute("EXPLAIN " + sql, [vec]).fetchall()[0][1]
                assert "HNSW_INDEX_SCAN" in plan_text.upper().replace(" ", "_"), f"{q['id']}: HNSW index not used"
                row[name] = {"pool": pool, "recall": len(set(pred) & set(truth)) / len(truth),
                             "fill": len(pred) / len(truth), "ndcg": ndcg(pred, truth), "ms": round(ms, 1)}
            rows.append(row)
    finally:
        conn.close()

    def summarize(subset):
        out = {"queries": len(subset)}
        for name in ("fixed_150", "adaptive"):
            out[name] = {m: round(float(np.mean([r[name][m] for r in subset])), 3) for m in ("recall", "fill", "ndcg")}
            out[name]["p50_ms"] = round(float(np.percentile([r[name]["ms"] for r in subset], 50)), 1)
        return out

    buckets = {
        f"plan_b_range (N > {LIMIT:,})": [r for r in rows if r["candidates"] > LIMIT],
        "5,000 < N <= 20,000": [r for r in rows if 5000 < r["candidates"] <= 20000],
        "N <= 5,000": [r for r in rows if r["candidates"] <= 5000],
        "all": rows,
    }
    summary = {label: summarize(subset) for label, subset in buckets.items() if subset}
    for label, s in summary.items():
        print(f"\n{label}: {s['queries']} queries")
        for name in ("fixed_150", "adaptive"):
            m = s[name]
            print(f"  {name:10s} recall@{K}={m['recall']:.3f}  fill@{K}={m['fill']:.3f}  ndcg@{K}={m['ndcg']:.3f}  p50={m['p50_ms']}ms")

    out = Path("benchmarks/retrieval_benchmark_results.json")
    out.write_text(json.dumps({"k": K, "summary": summary, "queries": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    run()
