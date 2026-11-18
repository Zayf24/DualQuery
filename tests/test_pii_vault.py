"""Tests for the local PII vault: redaction patterns, Luhn validation, and keyed pseudonyms."""

import pytest

from dualquery.security.pii_vault import PIIVault, luhn_checksum_valid


@pytest.fixture
def vault():
    return PIIVault(session_salt="test-salt")


@pytest.mark.parametrize("text,token,kind", [
    ("mail me at ana.souza@gmail.com please", "[EMAIL_REDACTED]", "email"),
    ("card 4111 1111 1111 1111 was charged", "[CARD_REDACTED]", "credit_card"),
    ("meu CPF 123.456.789-09", "[CPF_REDACTED]", "cpf"),
    ("CNPJ 12.345.678/0001-90 da loja", "[CNPJ_REDACTED]", "cnpj"),
    ("ligue (11) 98765-4321 hoje", "[PHONE_REDACTED]", "phone"),
    ("call +1 415-555-0132 now", "[PHONE_REDACTED]", "phone"),
    ("entregue na Rua Augusta, 1500 ontem", "[ADDRESS_REDACTED]", "address"),
])
def test_each_pii_type_is_redacted(vault, text, token, kind):
    res = vault.scrub_text(text)
    assert token in res.sanitized_text
    assert res.redaction_details.get(kind, 0) >= 1 and res.redactions_count >= 1


def test_card_number_failing_luhn_is_not_called_a_card(vault):
    assert luhn_checksum_valid("4111111111111111")
    assert not luhn_checksum_valid("4111111111111112")
    assert "[CARD_REDACTED]" not in vault.scrub_text("order ref 4111 1111 1111 1112").sanitized_text


def test_phone_counted_once(vault):
    res = vault.scrub_text("call 555-019-2834")
    assert res.redaction_details == {"phone": 1} and res.redactions_count == 1


@pytest.mark.parametrize("text", ["", "great product, arrived on time", "nota 10"])
def test_clean_text_is_unchanged(vault, text):
    res = vault.scrub_text(text)
    assert res.sanitized_text == text and res.redactions_count == 0


def test_pseudonyms_are_deterministic_keyed_and_opaque():
    a, b = PIIVault(session_salt="s1"), PIIVault(session_salt="s2")
    tok = a.tokenize_identifier("e481f51cbdc54678b7cc49136f2d6af7", "ord")
    assert tok == a.tokenize_identifier("e481f51cbdc54678b7cc49136f2d6af7", "ORD")
    assert tok != b.tokenize_identifier("e481f51cbdc54678b7cc49136f2d6af7", "ORD")
    assert tok.startswith("[ORD_") and len(tok) == len("[ORD_12345678]") and "e481" not in tok
    assert a.tokenize_identifier("", "ORD") == ""
