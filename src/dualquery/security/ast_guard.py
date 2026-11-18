"""
DualQuery Zero-Trust SQLGlot AST Security Guard.
Deterministic 5-gate compiler that validates, sanitizes, and rewrites SQL queries
at the Abstract Syntax Tree (AST) level before database execution.
"""

from typing import Set, Optional, List
import logging
import sqlglot
from sqlglot import exp
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ASTSecurityViolation(Exception):
    """Base exception for all AST security policy violations."""
    def __init__(self, message: str, gate: int, violation_type: str):
        super().__init__(message)
        self.gate = gate
        self.violation_type = violation_type


class ReadOnlyViolation(ASTSecurityViolation):
    """Gate 1: Statement is not a pure read-only SELECT."""
    def __init__(self, message: str):
        super().__init__(message, gate=1, violation_type="READ_ONLY_VIOLATION")


class FilesystemFunctionViolation(ASTSecurityViolation):
    """Gate 2: Query attempts to call filesystem or network table functions."""
    def __init__(self, message: str):
        super().__init__(message, gate=2, violation_type="FILESYSTEM_FUNCTION_VIOLATION")


class UnauthorizedTableViolation(ASTSecurityViolation):
    """Gate 3: Query references tables not permitted for the user's role."""
    def __init__(self, message: str):
        super().__init__(message, gate=3, violation_type="UNAUTHORIZED_TABLE_VIOLATION")


class RestrictedColumnViolation(ASTSecurityViolation):
    """Gate 4: Query accesses restricted PII or sensitive columns."""
    def __init__(self, message: str):
        super().__init__(message, gate=4, violation_type="RESTRICTED_COLUMN_VIOLATION")


class SecurityPolicy(BaseModel):
    """Role-Based Access Control and Row-Level Security Policy."""
    role_name: str = "analyst"
    allowed_tables: Set[str] = Field(
        default_factory=lambda: {
            "orders",
            "customers",
            "order_payments",
            "order_items",
            "products",
            "order_reviews"
        }
    )
    blocked_pii_columns: Set[str] = Field(
        default_factory=lambda: {
            "customer_unique_id",
            "cpf",
            "cnpj",
            "email",
            "phone",
            "password",
            "tax_id",
            "credit_card",
            "credit_card_number",
            "ssn",
            "secret"
        }
    )
    max_limit: int = Field(default=1000, description="Hard cap on returned rows")
    rls_predicate: Optional[str] = Field(default=None, description="Mandatory Row-Level Security predicate")


# Pre-configured Enterprise Role Policies
ANALYST_POLICY = SecurityPolicy(
    role_name="analyst",
    allowed_tables={"orders", "customers", "order_payments", "order_items", "products", "order_reviews"},
    max_limit=1000
)

SUPPORT_AGENT_POLICY = SecurityPolicy(
    role_name="support_agent",
    # Support agents can inspect orders, customer records, items, and reviews, but NOT financial payments
    allowed_tables={"orders", "customers", "order_items", "products", "order_reviews"},
    max_limit=500
)

SELLER_RESTRICTED_POLICY = SecurityPolicy(
    role_name="seller_restricted",
    # Sellers cannot access payment ledgers or global customer data directly
    allowed_tables={"orders", "order_items", "products", "order_reviews"},
    # Demo tenant: the busiest Olist seller (1,854 orders). Each seller deployment sets its own id.
    rls_predicate="order_items.seller_id = '6560211a19b47992c3666cc44a7e94c0'",
    max_limit=500
)


class ASTGuard:
    """
    5-Gate AST Security Firewall.
    Parses arbitrary SQL strings into SQLGlot ASTs and enforces deterministic security policies.
    """

    # Forbidden DDL / DML / Admin expression types for Gate 1
    FORBIDDEN_EXPRESSIONS = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Command,
        exp.Pragma,
        exp.Transaction,
        exp.Commit,
        exp.Rollback,
        exp.Create,
        exp.Attach,
        exp.Detach,
        exp.Copy,
        exp.Merge,
        exp.TruncateTable,
    )

    # Forbidden table functions for Gate 2 (filesystem / network sandbox escape)
    FORBIDDEN_TABLE_FUNCTIONS = {
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "scan_csv",
        "scan_parquet",
        "parquet_scan",
        "csv_scan",
        "glob",
        "query",
        "httpfs",
        "arrow_scan",
        "duckdb_settings",
        "duckdb_secrets",
        "getenv",
        "current_setting",
    }

    def __init__(self, default_policy: Optional[SecurityPolicy] = None):
        self.default_policy = default_policy or ANALYST_POLICY

    def validate_and_rewrite(self, sql: str, policy: Optional[SecurityPolicy] = None) -> str:
        """
        Validate incoming SQL against 5 security gates and return hardened SQL.
        Raises ASTSecurityViolation on any policy failure.
        """
        active_policy = policy or self.default_policy

        if not sql or not sql.strip():
            raise ReadOnlyViolation("Empty SQL query is invalid.")

        # Parse with DuckDB dialect
        try:
            statements = sqlglot.parse(sql, read="duckdb")
        except Exception as e:
            raise ASTSecurityViolation(f"SQL Syntax parsing failed: {e}", gate=0, violation_type="SYNTAX_ERROR")

        if not statements or statements[0] is None:
            raise ReadOnlyViolation("Failed to parse SQL into valid statements.")

        # Gate 1A: Strictly enforce single statement (blocks multi-statement injection attacks)
        if len(statements) > 1:
            raise ReadOnlyViolation(
                f"Multi-statement execution is strictly blocked. Detected {len(statements)} statements."
            )

        ast = statements[0]

        # Gate 1B: Root must be a SELECT (or SetOperation / Union of SELECTs)
        if isinstance(ast, exp.Block):
            raise ReadOnlyViolation("Block execution is strictly forbidden.")

        if not isinstance(ast, (exp.Select, exp.Union)):
            raise ReadOnlyViolation(
                f"Root AST node must be SELECT or UNION, got '{type(ast).__name__}'."
            )

        # Gate 1C: Traverse tree for any embedded mutation / administrative statements
        for forbidden_cls in self.FORBIDDEN_EXPRESSIONS:
            match = ast.find(forbidden_cls)
            if match:
                raise ReadOnlyViolation(
                    f"Forbidden statement type detected: '{forbidden_cls.__name__}'."
                )

        # Gate 2: Block File-system, Network, and Host Table Functions
        # Check dedicated SQLGlot table function types
        if ast.find(exp.ReadCSV) or ast.find(exp.ReadParquet):
            raise FilesystemFunctionViolation("External file table functions (read_csv/read_parquet) are forbidden.")

        for func in ast.find_all(exp.Func, exp.Anonymous):
            func_name = func.name.lower() if func.name else ""
            if func_name in self.FORBIDDEN_TABLE_FUNCTIONS:
                raise FilesystemFunctionViolation(
                    f"Unauthorized filesystem/network function call: '{func_name}()'."
                )

        # Also inspect any Table nodes where the target is a function call rather than a plain identifier
        for table in ast.find_all(exp.Table):
            if isinstance(table.this, (exp.Func, exp.Anonymous, exp.ReadCSV, exp.ReadParquet)):
                func_name = getattr(table.this, "name", str(table.this)).lower()
                raise FilesystemFunctionViolation(
                    f"Direct table function invocation is blocked: '{func_name}'."
                )

        # Gate 3: Table-Level RBAC & CTE Smuggling Defense
        # 1. Identify all CTE alias names
        cte_aliases: Set[str] = {cte.alias_or_name.lower() for cte in ast.find_all(exp.CTE) if cte.alias_or_name}
        
        # 2. Extract all referenced physical table names (excluding CTE aliases)
        referenced_tables: Set[str] = set()
        for t in ast.find_all(exp.Table):
            if isinstance(t.this, exp.Identifier):
                t_name = t.name.lower()
                if t_name not in cte_aliases:
                    referenced_tables.add(t_name)

        # Also extract table qualifiers from referenced columns (e.g. order_payments.payment_value)
        for c in ast.find_all(exp.Column):
            if c.table:
                c_tbl = c.table.lower()
                if c_tbl not in cte_aliases:
                    referenced_tables.add(c_tbl)

        # 3. Verify against allowed tables
        allowed = {t.lower() for t in active_policy.allowed_tables}
        unauthorized = referenced_tables - allowed
        if unauthorized:
            raise UnauthorizedTableViolation(
                f"Role '{active_policy.role_name}' denied access to tables: {sorted(list(unauthorized))}."
            )

        # Gate 4: Column-Level PII Shield
        referenced_cols = {c.name.lower() for c in ast.find_all(exp.Column) if c.name}
        blocked_pii = {c.lower() for c in active_policy.blocked_pii_columns}
        violating_cols = referenced_cols & blocked_pii
        if violating_cols:
            raise RestrictedColumnViolation(
                f"Access to restricted PII columns blocked: {sorted(list(violating_cols))}."
            )

        # Gate 5: Programmatic AST Rewriting (RLS Predicate & Hard LIMIT Injection)
        # 5A: Inject Row-Level Security predicate if defined
        if active_policy.rls_predicate:
            try:
                rls_cond = sqlglot.parse_one(active_policy.rls_predicate, read="duckdb")
                ast = ast.where(rls_cond, append=True)
            except Exception as e:
                logger.error(f"Failed to inject RLS predicate '{active_policy.rls_predicate}': {e}")
                raise ASTSecurityViolation(
                    f"RLS predicate injection failed: {e}", gate=5, violation_type="RLS_INJECTION_FAILURE"
                )

        # 5B: Enforce Hard LIMIT Cap
        limit_arg = ast.args.get("limit")
        if limit_arg is None:
            ast = ast.limit(active_policy.max_limit)
        else:
            try:
                current_limit = int(limit_arg.expression.this)
                if current_limit > active_policy.max_limit:
                    ast = ast.limit(active_policy.max_limit)
            except (ValueError, AttributeError):
                ast = ast.limit(active_policy.max_limit)

        # Compile and return sanitized DuckDB SQL
        return ast.sql(dialect="duckdb")
