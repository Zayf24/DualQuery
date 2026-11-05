"""
DualQuery Pre-SQL Entity Literal Grounder.
Resolves informal user terms (e.g., 'São Paulo', 'credit card', 'cosmetics')
into exact, verified database literals ('SP', 'credit_card', 'health_beauty')
before SQL query compilation.
"""

import json
import logging
import unicodedata
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel, Field
import rapidfuzz
from rapidfuzz import fuzz

from dualquery.config import settings

logger = logging.getLogger(__name__)


class GroundedLiteral(BaseModel):
    """Represents a successfully resolved database enum literal."""
    domain: str = Field(description="Database column domain (e.g., customer_state, payment_type)")
    raw_input: str = Field(description="Original user term")
    canonical_literal: str = Field(description="Exact value as physically stored in DuckDB")
    confidence: float = Field(description="Matching confidence score between 0.0 and 1.0")
    method: str = Field(description="Resolution method: 'exact', 'alias', 'fuzzy', or 'semantic'")


class LiteralGrounder:
    """
    Two-Tier Pre-SQL Entity Literal Grounder:
    - Tier 1: Deterministic normalization, Brazilian geography alias table, and RapidFuzz WRatio (<1ms)
    - Tier 2: Semantic embedding cosine distance for cross-lingual / conceptual categories
    """

    # Brazilian state geographical aliases
    STATE_ALIASES: Dict[str, str] = {
        "sao paulo": "SP", "rio de janeiro": "RJ", "minas gerais": "MG",
        "rio grande do sul": "RS", "parana": "PR", "santa catarina": "SC",
        "bahia": "BA", "distrito federal": "DF", "brasilia": "DF",
        "espirito santo": "ES", "goias": "GO", "maranhao": "MA",
        "mato grosso": "MT", "mato grosso do sul": "MS", "para": "PA",
        "paraiba": "PB", "pernambuco": "PE", "piaui": "PI",
        "rio grande do norte": "RN", "rondonia": "RO", "roraima": "RR",
        "sergipe": "SE", "tocantins": "TO", "acre": "AC", "alagoas": "AL",
        "amapa": "AP", "amazonas": "AM", "ceara": "CE"
    }

    # Common colloquial payment aliases
    PAYMENT_ALIASES: Dict[str, str] = {
        "credit card": "credit_card", "credit": "credit_card", "cc": "credit_card",
        "debit card": "debit_card", "debit": "debit_card",
        "voucher": "voucher", "coupon": "voucher", "gift card": "voucher",
        "boleto": "boleto", "bank slip": "boleto", "bank invoice": "boleto"
    }

    # Order status aliases
    STATUS_ALIASES: Dict[str, str] = {
        "delivered": "delivered", "completed": "delivered", "arrived": "delivered",
        "canceled": "canceled", "cancelled": "canceled", "refunded": "canceled",
        "shipped": "shipped", "in transit": "shipped", "dispatched": "shipped",
        "invoiced": "invoiced", "processing": "processing"
    }

    def __init__(self, dictionary_path: Optional[Path] = None, embedder=None):
        self.dict_path = dictionary_path or (settings.DATA_DIR / "literal_dictionary.json")
        self.embedder = embedder
        self.domains: Dict[str, List[str]] = self._load_dictionary()

    def _load_dictionary(self) -> Dict[str, List[str]]:
        """Load distinct values extracted by literal_indexer."""
        if not self.dict_path.exists():
            logger.warning(f"Literal dictionary not found at {self.dict_path}. Initializing empty.")
            return {}
        with open(self.dict_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Strip accents, lowercase, and clean whitespace."""
        text = text.strip().lower()
        # Remove accents: 'São Paulo' -> 'Sao Paulo'
        nfkd = unicodedata.normalize('NFKD', text)
        return "".join([c for c in nfkd if not unicodedata.combining(c)])

    def resolve_term(self, term: str, target_domain: Optional[str] = None) -> Optional[GroundedLiteral]:
        """
        Resolve an individual input term into its grounded canonical literal.
        If target_domain is specified, search is restricted to that domain.
        """
        norm_term = self.normalize_text(term)
        if not norm_term:
            return None

        # 1. Check State Aliases
        if target_domain in (None, "customer_state"):
            if norm_term.upper() in self.domains.get("customer_state", []):
                return GroundedLiteral(
                    domain="customer_state",
                    raw_input=term,
                    canonical_literal=norm_term.upper(),
                    confidence=1.0,
                    method="exact"
                )
            if norm_term in self.STATE_ALIASES:
                return GroundedLiteral(
                    domain="customer_state",
                    raw_input=term,
                    canonical_literal=self.STATE_ALIASES[norm_term],
                    confidence=0.99,
                    method="alias"
                )

        # 2. Check Payment Aliases
        if target_domain in (None, "payment_type"):
            if norm_term in self.PAYMENT_ALIASES:
                return GroundedLiteral(
                    domain="payment_type",
                    raw_input=term,
                    canonical_literal=self.PAYMENT_ALIASES[norm_term],
                    confidence=0.98,
                    method="alias"
                )

        # 3. Check Order Status Aliases
        if target_domain in (None, "order_status"):
            if norm_term in self.STATUS_ALIASES:
                return GroundedLiteral(
                    domain="order_status",
                    raw_input=term,
                    canonical_literal=self.STATUS_ALIASES[norm_term],
                    confidence=0.98,
                    method="alias"
                )

        # 4. Search Domains using RapidFuzz
        # Note: We exclude 'customer_state' from generic fuzzy matching because 2-letter state codes
        # (e.g. 'ES', 'PA') produce false substring matches against words like 'watches' or 'parties'.
        if target_domain:
            search_domains = [target_domain]
        else:
            search_domains = [d for d in self.domains.keys() if d != "customer_state"]
        best_match: Optional[Tuple[str, str, float]] = None  # (domain, literal, score)

        for domain in search_domains:
            candidates = self.domains.get(domain, [])
            if not candidates:
                continue

            # Check exact match
            for cand in candidates:
                cand_norm = self.normalize_text(cand.replace("_", " "))
                if norm_term == cand_norm or norm_term == cand.lower():
                    return GroundedLiteral(
                        domain=domain,
                        raw_input=term,
                        canonical_literal=cand,
                        confidence=1.0,
                        method="exact"
                    )

            # RapidFuzz WRatio match
            # Compare normalized strings with underscores treated as spaces
            cand_map = {self.normalize_text(c.replace("_", " ")): c for c in candidates}
            result = rapidfuzz.process.extractOne(
                norm_term,
                list(cand_map.keys()),
                scorer=fuzz.WRatio
            )
            if result:
                matched_key, score, _ = result
                conf = score / 100.0
                if conf >= 0.85:
                    if best_match is None or conf > best_match[2]:
                        best_match = (domain, cand_map[matched_key], conf)

        if best_match:
            return GroundedLiteral(
                domain=best_match[0],
                raw_input=term,
                canonical_literal=best_match[1],
                confidence=best_match[2],
                method="fuzzy"
            )

        # 5. Tier 2: Semantic Category Match via Embeddings (if embedder provided)
        if self.embedder and target_domain in (None, "product_category_name_english"):
            return self._resolve_semantic_category(term)

        return None

    def _resolve_semantic_category(self, term: str) -> Optional[GroundedLiteral]:
        """Use embedding similarity to resolve conceptual category names (e.g., 'cosmetics' -> 'health_beauty')."""
        categories = self.domains.get("product_category_name_english", [])
        if not categories or not self.embedder:
            return None

        import numpy as np

        query_vec = np.array(self.embedder.embed_query(term), dtype=np.float32)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-9)

        # Generate representations for candidate categories
        cand_texts = [c.replace("_", " ") for c in categories]
        cand_vecs = np.array(self.embedder.embed_documents(cand_texts), dtype=np.float32)
        cand_norms = cand_vecs / (np.linalg.norm(cand_vecs, axis=1, keepdims=True) + 1e-9)

        similarities = np.dot(cand_norms, query_norm)
        best_idx = int(np.argmax(similarities))
        best_sim = float(similarities[best_idx])

        if best_sim >= 0.70:
            return GroundedLiteral(
                domain="product_category_name_english",
                raw_input=term,
                canonical_literal=categories[best_idx],
                confidence=round(best_sim, 4),
                method="semantic"
            )

        return None

    def resolve_terms(self, terms: List[str]) -> Dict[str, GroundedLiteral]:
        """Resolve a batch of terms and return a mapped dictionary."""
        grounded = {}
        for t in terms:
            res = self.resolve_term(t)
            if res:
                grounded[t] = res
        return grounded
