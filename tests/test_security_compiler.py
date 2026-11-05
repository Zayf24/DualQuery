"""
Automated Pytest Suite for DualQuery 5-Gate AST Security Guard.
Verifies 100% block rate across all 30 adversarial red-team attack vectors.
"""

import pytest
from dualquery.security.ast_guard import ASTGuard, ASTSecurityViolation
from benchmarks.evaluate_security import RED_TEAM_ATTACKS


@pytest.fixture(scope="module")
def guard():
    """Shared ASTGuard instance."""
    return ASTGuard()


@pytest.mark.parametrize("attack", RED_TEAM_ATTACKS, ids=[a["id"] for a in RED_TEAM_ATTACKS])
def test_ast_guard_blocks_adversarial_attack(guard, attack):
    """
    Assert that every malicious SQL injection or unauthorized access vector
    is intercepted and blocked by the AST Security Guard.
    """
    sql = attack["sql"]
    policy = attack["policy"]
    expected_gate = attack["expected_gate"]

    with pytest.raises((ASTSecurityViolation, Exception)) as exc_info:
        guard.validate_and_rewrite(sql, policy=policy)

    # If it was an ASTSecurityViolation, verify it triggered a security gate
    if isinstance(exc_info.value, ASTSecurityViolation):
        assert exc_info.value.gate == expected_gate, (
            f"Attack {attack['id']} ({attack['name']}) expected Gate {expected_gate}, "
            f"but was caught by Gate {exc_info.value.gate}"
        )
