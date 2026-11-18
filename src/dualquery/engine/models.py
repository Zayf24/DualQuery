"""
DualQuery Core Engine Data Models.
Decoupled model definitions to prevent circular imports across security and execution.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class QueryCitation(BaseModel):
    """An audited review citation returned as evidence."""
    review_id: str
    order_id: str
    review_score: int
    review_comment: str
    score_or_distance: float


class QueryExecutionResult(BaseModel):
    """Audited execution result with full telemetry."""
    plan_type: str = Field(description="Plan A (exact scan) or Plan B (HNSW + BM25 RRF)")
    estimated_candidates: int = Field(description="Orders matching the validated SQL filter (N_cand)")
    probe_latency_ms: float = Field(description="Latency of pre-flight probe in ms")
    engine_latency_ms: float = Field(description="Latency of the retrieval SQL in ms")
    total_latency_ms: float = Field(description="End-to-end engine turn latency in ms")
    aggregates: Dict[str, Any] = Field(default_factory=dict, description="Metrics over every order matching the SQL filter")
    matched_orders: int = Field(default=0, description="Filtered orders with a review in the top match_top_percent most similar to the semantic query")
    matched_aggregates: Dict[str, Any] = Field(default_factory=dict, description="Metrics over matched_orders only")
    citations: List[QueryCitation] = Field(default_factory=list, description="Top evidence citations")
