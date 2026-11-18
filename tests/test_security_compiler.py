"""
AST Security Guard tests: the 30-attack red-team suite must be stopped by the expected gate
(not by an incidental crash), legitimate queries must pass, and rewrites must be correct.
"""

import pytest
from dualquery.security.ast_guard import ASTGuard, ASTSecurityViolation, ANALYST_POLICY, SecurityPolicy
from benchmarks.evaluate_security import RED_TEAM_ATTACKS


@pytest.fixture(scope="module")
def guard():
    return ASTGuard()


@pytest.mark.parametrize("attack", RED_TEAM_ATTACKS, ids=[a["id"] for a in RED_TEAM_ATTACKS])
def test_ast_guard_blocks_adversarial_attack(guard, attack):
    with pytest.raises(ASTSecurityViolation) as exc_info:
        guard.validate_and_rewrite(attack["sql"], policy=attack["policy"])
    assert exc_info.value.gate == attack["expected_gate"], (
        f"{attack['id']} ({attack['name']}) expected Gate {attack['expected_gate']}, got Gate {exc_info.value.gate}"
    )


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders WHERE length(getenv('HOME')) > 0",
    "SELECT * FROM orders WHERE orders.order_id IN (SELECT current_setting('home_directory'))",
])
def test_host_introspection_functions_are_blocked(guard, sql):
    with pytest.raises(ASTSecurityViolation) as exc_info:
        guard.validate_and_rewrite(sql, policy=ANALYST_POLICY)
    assert exc_info.value.gate == 2


@pytest.mark.parametrize("sql", [
    "SELECT * FROM orders JOIN customers ON orders.customer_id = customers.customer_id WHERE customers.customer_state = 'RJ'",
    "SELECT * FROM orders WHERE orders.order_id IN (SELECT order_reviews.order_id FROM order_reviews WHERE order_reviews.review_score <= 2)",
    "SELECT COUNT(DISTINCT orders.order_id), ROUND(SUM(order_payments.payment_value), 2) FROM orders JOIN order_payments ON orders.order_id = order_payments.order_id",
])
def test_legitimate_queries_pass(guard, sql):
    assert guard.validate_and_rewrite(sql, policy=ANALYST_POLICY).startswith("SELECT")


def test_rls_is_and_ed_around_whole_user_predicate(guard):
    policy = SecurityPolicy(role_name="t", rls_predicate="order_items.seller_id = 'S1'")
    out = guard.validate_and_rewrite("SELECT * FROM orders WHERE orders.a = 1 OR orders.b = 2", policy=policy)
    assert "WHERE (orders.a = 1 OR orders.b = 2) AND order_items.seller_id = 'S1'" in out


@pytest.mark.parametrize("sql,expected_limit", [
    ("SELECT * FROM orders", "LIMIT 1000"),
    ("SELECT * FROM orders LIMIT 5", "LIMIT 5"),
    ("SELECT * FROM orders LIMIT 99999", "LIMIT 1000"),
])
def test_limit_is_capped(guard, sql, expected_limit):
    assert guard.validate_and_rewrite(sql, policy=ANALYST_POLICY).endswith(expected_limit)


def test_empty_and_unparseable_sql_are_rejected(guard):
    for sql in ("", "   "):
        with pytest.raises(ASTSecurityViolation):
            guard.validate_and_rewrite(sql)
    with pytest.raises(ASTSecurityViolation):
        guard.validate_and_rewrite("SELECT FROM WHERE (((")
