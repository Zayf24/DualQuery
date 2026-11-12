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


class Settings(BaseModel):
    # Base Directories
    PROJECT_ROOT: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)
    DATA_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data")
    SQL_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "sql")
    
    # DuckDB Database File
    DUCKDB_PATH: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent / "data" / "olist.duckdb")
    
    # Embedding Configuration
    # Unified dimension: 768 (EmbeddingGemma 2, Nomic-Embed-v1.5, or BGE-Base)
    EMBEDDING_DIM: int = 768
    PRIMARY_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    FALLBACK_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 256
    
    # Optimizer Thresholds
    # Empirically measured inflection threshold (benchmarks/evaluate_optimizer.py):
    # SIMD vector scan maintains sub-400ms latency and 100% recall up to ~5,000 candidates.
    # Beyond 5,000 candidates, Plan B (HNSW + BM25 RRF) is engaged.
    PLAN_A_CARDINALITY_LIMIT: int = 5000


settings = Settings()
