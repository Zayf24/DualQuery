"""
DualQuery SQL Expression Normalizer and Join Resolver.
Canonicalizes table aliases and builds minimal deterministic JOIN trees for DuckDB.
"""

import re
from typing import List, Union


ALIAS_REPLACEMENTS = [
    (r'\bc\.', 'customers.'),
    (r'\bp\.', 'order_payments.'),
    (r'\bi\.', 'order_items.'),
    (r'\bprod\.', 'products.'),
    (r'\bo\.', 'orders.'),
    (r'\br\.', 'order_reviews.'),
]


def normalize_sql_expression(expr: str) -> str:
    """Normalize short table aliases (c., p., o., i., prod., r.) to canonical table names."""
    if not expr:
        return ""
    normalized = expr
    for pattern, repl in ALIAS_REPLACEMENTS:
        normalized = re.sub(pattern, repl, normalized)
    return normalized


def get_required_joins(expressions: Union[str, List[str]], base_table: str = "orders") -> str:
    """
    Determine minimal JOIN clauses required to satisfy table references in SQL expressions.
    Ensures safe join order (e.g. order_items before products) without duplicate joins.
    """
    if isinstance(expressions, str):
        combined = expressions.lower()
    else:
        combined = " ".join(expressions).lower()

    joins: List[str] = []

    # Customers
    if "customers." in combined or "customer_" in combined:
        joins.append("JOIN customers ON orders.customer_id = customers.customer_id")

    # Order Payments
    if "order_payments." in combined or "payment_" in combined:
        joins.append("JOIN order_payments ON orders.order_id = order_payments.order_id")

    # Order Items (or Products which depend on Order Items)
    needs_items = (
        "order_items." in combined
        or "freight_value" in combined
        or "seller_id" in combined
        or "price" in combined
    )
    needs_products = "products." in combined or "product_" in combined

    if needs_items or needs_products:
        joins.append("JOIN order_items ON orders.order_id = order_items.order_id")

    # Products
    if needs_products:
        joins.append("JOIN products ON order_items.product_id = products.product_id")

    # Order Reviews (if base_table is orders and reviews are referenced)
    if base_table == "orders" and ("order_reviews." in combined or "review_" in combined):
        joins.append("JOIN order_reviews ON orders.order_id = order_reviews.order_id")

    return "\n            ".join(joins)

