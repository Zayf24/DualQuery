"""
DualQuery Global Configuration and Settings.
Centralizes paths, embedding configurations, and execution thresholds.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Keep heavy model caches on D: drive on Windows workstations when present,
# while remaining portable across macOS/Linux/single-drive setups.
if os.name == "nt" and Path("D:\\.cache").exists():
    os.environ.setdefault("HF_HOME", "D:\\.cache\\huggingface")
    os.environ.setdefault("UV_CACHE_DIR", "D:\\.cache\\uv")

_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseModel):
    # Base Directories
    DATA_DIR: Path = _ROOT / "data"
    SQL_DIR: Path = _ROOT / "sql"

    # DuckDB Database File (override with DUALQUERY_DB_PATH to serve another Olist build)
    DUCKDB_PATH: Path = Field(
        default_factory=lambda: Path(os.environ.get("DUALQUERY_DB_PATH", _ROOT / "data" / "olist.duckdb"))
    )

    # Embedding Configuration (BAAI/bge-base-en-v1.5, 768 dimensions)
    EMBEDDING_DIM: int = 768
    PRIMARY_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 256

    # Optimizer Threshold: filters matching <= this many orders run Plan A (exact scan, 100% recall),
    # broader filters run Plan B (HNSW + BM25 RRF). 30,000 is the measured latency crossover on the
    # 99k-order Olist build (benchmarks/evaluate_optimizer.py): exact scan is faster up to ~20k orders,
    # Plan B is faster from ~42k.
    PLAN_A_CARDINALITY_LIMIT: int = 30000


settings = Settings()
