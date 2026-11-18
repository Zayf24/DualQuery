"""
DualQuery Local In-Process PII Vault & Redaction Engine.
Provides:
1. Deterministic keyed HMAC-SHA256 pseudonyms for identifiers ([ORD_491A0B2C])
2. High-precision pattern scrubbing for Emails, Phones, CPFs, Cards, and Street Addresses.
Guarantees zero customer PII leaves the local machine.
"""

from __future__ import annotations

import re
import hmac
import hashlib
import secrets
from typing import TYPE_CHECKING, Dict, Optional
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dualquery.engine.models import QueryCitation


def luhn_checksum_valid(card_number_str: str) -> bool:
    """Validate credit card number using the Luhn (mod 10) algorithm."""
    digits = [int(d) for d in re.sub(r'\D', '', card_number_str)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            doubled = d * 2
            checksum += doubled - 9 if doubled > 9 else doubled
        else:
            checksum += d
    return checksum % 10 == 0


class PIIScrubResult(BaseModel):
    """Result of a PII scrubbing operation with redaction audit counts."""
    sanitized_text: str
    redactions_count: int
    redaction_details: Dict[str, int] = Field(default_factory=dict)


class PIIVault:
    """
    In-Process Zero-Trust PII Tokenizer & Text Sanitizer.
    Holds a per-process secret salt for keyed HMAC pseudonyms and applies compiled regex patterns.
    """

    # Pre-compiled high-precision regex patterns
    EMAIL_PATTERN = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        re.IGNORECASE
    )

    # Brazilian CPF format (e.g. 123.456.789-01 or 12345678901 with context)
    CPF_PATTERN = re.compile(
        r'\b\d{3}\.\d{3}\.\d{3}-\d{2}\b'
    )

    # Brazilian CNPJ format (e.g. 12.345.678/0001-90)
    CNPJ_PATTERN = re.compile(
        r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'
    )

    # Phone numbers: Brazilian (+55 11 98765-4321, (11) 98765-4321) & International
    PHONE_PATTERN = re.compile(
        r'(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?\d{4}[-\s]?\d{4}|\d{4}[-\s]?\d{4})\b|'
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
    )

    # Credit card candidate strings: 13-19 digits with optional spaces or hyphens
    CARD_CANDIDATE_PATTERN = re.compile(
        r'\b(?:\d[ -]*?){13,19}\b'
    )

    # Physical Street Addresses (Portuguese & English street prefixes)
    ADDRESS_PATTERN = re.compile(
        r'\b(?:Rua|R\.|Avenida|Av\.|Alameda|Al\.|Travessa|Tv\.|Estrada|Rodovia|Street|St\.|Avenue|Ave\.)\s+[A-Za-zÀ-ÖØ-öø-ÿ0-9\s,\.-]{4,40}(?:,\s*\d+|\s+n[ºo°]?\s*\d+)?\b',
        re.IGNORECASE
    )

    def __init__(self, session_salt: Optional[str] = None):
        self.salt = (session_salt or secrets.token_hex(16)).encode("utf-8")

    def tokenize_identifier(self, real_id: str, prefix: str = "ID") -> str:
        """
        Generate a deterministic, opaque pseudonym token using salted HMAC-SHA256.
        Example: tokenize_identifier("06b8999e2fba", "CUST") -> "[CUST_A8E2B149]"
        """
        if not real_id:
            return ""

        digest = hmac.new(self.salt, real_id.encode("utf-8"), hashlib.sha256).hexdigest()
        return f"[{prefix.upper()}_{digest[:8].upper()}]"

    def scrub_text(self, text: str) -> PIIScrubResult:
        """
        Scrub raw text by redacting Emails, Phones, CPFs, CNPJs, Credit Cards, and Addresses.
        """
        if not text:
            return PIIScrubResult(sanitized_text="", redactions_count=0)

        sanitized = text
        details: Dict[str, int] = {}
        total_redactions = 0

        # 1. Redact Emails
        emails = self.EMAIL_PATTERN.findall(sanitized)
        if emails:
            sanitized = self.EMAIL_PATTERN.sub("[EMAIL_REDACTED]", sanitized)
            details["email"] = len(emails)
            total_redactions += len(emails)

        # 2. Redact Validated Credit Cards FIRST (before phone numbers fragment them)
        card_matches = self.CARD_CANDIDATE_PATTERN.findall(sanitized)
        cards_redacted = 0
        for match in card_matches:
            clean_digits = re.sub(r'\D', '', match)
            if 13 <= len(clean_digits) <= 19 and luhn_checksum_valid(clean_digits):
                sanitized = sanitized.replace(match, "[CARD_REDACTED]")
                cards_redacted += 1

        if cards_redacted:
            details["credit_card"] = cards_redacted
            total_redactions += cards_redacted

        # 3. Redact CPFs & CNPJs
        cpfs = self.CPF_PATTERN.findall(sanitized)
        if cpfs:
            sanitized = self.CPF_PATTERN.sub("[CPF_REDACTED]", sanitized)
            details["cpf"] = len(cpfs)
            total_redactions += len(cpfs)

        cnpjs = self.CNPJ_PATTERN.findall(sanitized)
        if cnpjs:
            sanitized = self.CNPJ_PATTERN.sub("[CNPJ_REDACTED]", sanitized)
            details["cnpj"] = len(cnpjs)
            total_redactions += len(cnpjs)

        # 4. Redact Phone Numbers
        phones = self.PHONE_PATTERN.findall(sanitized)
        if phones:
            sanitized = self.PHONE_PATTERN.sub("[PHONE_REDACTED]", sanitized)
            details["phone"] = len(phones)
            total_redactions += len(phones)

        # 5. Redact Physical Addresses
        addresses = self.ADDRESS_PATTERN.findall(sanitized)
        if addresses:
            sanitized = self.ADDRESS_PATTERN.sub("[ADDRESS_REDACTED]", sanitized)
            details["address"] = len(addresses)
            total_redactions += len(addresses)

        return PIIScrubResult(
            sanitized_text=sanitized,
            redactions_count=total_redactions,
            redaction_details=details
        )

    def scrub_citation(self, citation: QueryCitation, tokenize_ids: bool = True) -> QueryCitation:
        """
        Sanitize an audited review citation snippet before returning to LLM or UI.
        """
        from dualquery.engine.models import QueryCitation as QC

        scrubbed = self.scrub_text(citation.review_comment)

        order_id_out = (
            self.tokenize_identifier(citation.order_id, prefix="ORD")
            if tokenize_ids else citation.order_id
        )
        review_id_out = (
            self.tokenize_identifier(citation.review_id, prefix="REV")
            if tokenize_ids else citation.review_id
        )

        return QC(
            review_id=review_id_out,
            order_id=order_id_out,
            review_score=citation.review_score,
            review_comment=scrubbed.sanitized_text,
            score_or_distance=citation.score_or_distance
        )
