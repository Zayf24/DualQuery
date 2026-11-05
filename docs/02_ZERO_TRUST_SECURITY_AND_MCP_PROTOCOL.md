# DualQuery (`dualquery-mcp`)
## Document 02: Zero-Trust Security & MCP Protocol (v2.0)

### 1. Compute-at-the-Edge Privacy Model

Standard database MCP servers violate enterprise governance by returning raw query rows to the host LLM. `dualquery-mcp` enforces a strict **Plan-Shipping, Aggregate-Returning** boundary:

1. **Zero Raw Table Egress:** The external LLM never receives raw transactional rows. All joins, filtering, and math (`SUM`, `AVG`, `STDDEV`, `QUANTILE`) happen locally inside DuckDB.
2. **Bounded Evidence Snippets:** When qualitative citations are requested to explain *why* a metric moved, the engine returns at most `top_k_citations <= 5` short text snippets after passing them through the local PII Scrubber.

---

### 2. MCP Server Tool Specifications (JSON-RPC 2.0)

`dualquery-mcp` exposes three strongly-typed tools over the Model Context Protocol:

#### Tool 1: `get_governed_catalog`
Returns only the tables, columns, foreign keys, and metrics authorized for the active role's security policy.

#### Tool 2: `resolve_entity_literals`
Maps user-supplied natural language strings to verified database enum values before SQL generation.

#### Tool 3: `execute_hybrid_query`
The primary execution workhorse. Accepts a declarative hybrid query plan rather than arbitrary unvalidated scripts:

```json
{
  "name": "execute_hybrid_query",
  "description": "Executes a cost-optimized hybrid relational SQL and semantic vector/BM25 query locally inside DuckDB, returning deterministic aggregations and PII-redacted evidence citations.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "sql_filter_query": {
        "type": "string",
        "description": "Read-only SELECT query returning candidate join keys (e.g., order_id) and relational attributes to aggregate."
      },
      "semantic_target": {
        "type": "object",
        "properties": {
          "table": {"type": "string", "enum": ["order_reviews", "support_tickets", "strategy_memos"]},
          "search_query": {"type": "string", "description": "Natural language concept to match via hybrid HNSW + BM25 RRF."},
          "min_similarity_threshold": {"type": "number", "default": 0.62},
          "top_k_candidates": {"type": "integer", "default": 250}
        },
        "required": ["table", "search_query"]
      },
      "aggregation_sql": {
        "type": "string",
        "description": "Outer SQL query computing SUM, AVG, COUNT, or GROUP BY over the joined CTE `hybrid_matches`."
      },
      "max_redacted_citations": {
        "type": "integer",
        "default": 5,
        "maximum": 10
      }
    },
    "required": ["sql_filter_query", "aggregation_sql"]
  }
}
```

---

### 3. Five-Stage `SQLGlot` AST Security Compiler

Every SQL fragment passed to `dualquery-mcp` is parsed into an Abstract Syntax Tree using `sqlglot.parse_one(sql, read="duckdb")` and validated through five deterministic gates:

```python
import sqlglot
from sqlglot import exp

class ASTSecurityViolation(Exception):
    pass

def enforce_ast_guardrails(sql: str, role_policy: dict) -> str:
    ast = sqlglot.parse_one(sql, read="duckdb")

    # Gate 1: Must be a read-only SELECT statement (blocks DROP, DELETE, UPDATE, INSERT, COPY, ATTACH, PRAGMA)
    if not isinstance(ast, exp.Select):
        raise ASTSecurityViolation(f"Root AST node must be SELECT, got {type(ast).__name__}")
    for forbidden in (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Alter, exp.Command, exp.Pragma):
        if ast.find(forbidden):
            raise ASTSecurityViolation(f"Forbidden statement type detected: {forbidden.__name__}")

    # Gate 2: Block file-system & network table functions (read_csv, read_parquet, httpfs)
    for func in ast.find_all(exp.Anonymous, exp.Func):
        if func.name.lower() in {"read_csv", "read_csv_auto", "read_parquet", "read_json", "glob", "query"}:
            raise ASTSecurityViolation(f"Unauthorized table/file function: {func.name}")

    # Gate 3: Table-level RBAC Whitelist Check (including CTEs and Subqueries)
    cte_aliases = {cte.alias_or_name.lower() for cte in ast.find_all(exp.CTE)}
    referenced_tables = {t.name.lower() for t in ast.find_all(exp.Table)} - cte_aliases
    unauthorized_tables = referenced_tables - set(role_policy["allowed_tables"])
    if unauthorized_tables:
        raise ASTSecurityViolation(f"Role '{role_policy['role']}' denied access to tables: {unauthorized_tables}")

    # Gate 4: Column-level PII Blocking
    referenced_cols = {c.name.lower() for c in ast.find_all(exp.Column)}
    blocked_cols = referenced_cols & set(role_policy["blocked_pii_columns"])
    if blocked_cols:
        raise ASTSecurityViolation(f"Direct selection/filtering on restricted PII columns blocked: {blocked_cols}")

    # Gate 5: Inject Mandatory Row-Level Security (RLS) Predicate & Hard LIMIT
    if role_policy.get("rls_predicate"):
        rls_cond = sqlglot.parse_one(role_policy["rls_predicate"], read="duckdb")
        ast = ast.where(rls_cond, append=True)

    if not ast.args.get("limit"):
        ast = ast.limit(1000)

    return ast.sql(dialect="duckdb")
```

---

### 4. Local PII Vault & Citation Scrubber

Before any unstructured review or ticket snippet is returned in `redacted_citations`, it passes through an in-process sanitization pipeline:

1. **Primary Key Opaque Tokenization:** Real `customer_unique_id` values are mapped via HMAC-SHA256 (salted per session) to reversible display tokens (`[CUST_491A]`) stored in a local in-memory lookup table.
2. **Regex + Pattern Redaction:** Strips Brazilian CPF/CNPJ tax IDs, Indian Aadhaar/PAN/GSTIN numbers, email addresses, phone/WhatsApp numbers, and street addresses before serialization.
