# DualQuery (`dualquery-mcp`)
## Document 04: Data Schemas, Benchmarks & Roadmap (v2.0)

### 1. Unified DuckDB Schema (`sql/01_init_schema.sql`)

```sql
INSTALL vss;
LOAD vss;
INSTALL fts;
LOAD fts;

SET hnsw_enable_experimental_persistence = true;

CREATE TABLE customers (
    customer_id VARCHAR PRIMARY KEY,
    customer_unique_id VARCHAR NOT NULL,
    customer_zip_code_prefix VARCHAR,
    customer_city VARCHAR,
    customer_state VARCHAR(2)
);

CREATE TABLE orders (
    order_id VARCHAR PRIMARY KEY,
    customer_id VARCHAR REFERENCES customers(customer_id),
    order_status VARCHAR,
    order_purchase_timestamp TIMESTAMP,
    order_delivered_customer_date TIMESTAMP,
    order_estimated_delivery_date TIMESTAMP,
    delivery_delay_days INTEGER GENERATED ALWAYS AS (
        date_diff('day', order_estimated_delivery_date, order_delivered_customer_date)
    ) VIRTUAL
);

CREATE TABLE order_payments (
    order_id VARCHAR REFERENCES orders(order_id),
    payment_sequential INTEGER,
    payment_type VARCHAR,
    payment_installments INTEGER,
    payment_value DECIMAL(10, 2)
);

CREATE TABLE products (
    product_id VARCHAR PRIMARY KEY,
    product_category_name VARCHAR,
    product_category_name_english VARCHAR,
    product_weight_g INTEGER
);

CREATE TABLE order_items (
    order_id VARCHAR REFERENCES orders(order_id),
    order_item_id INTEGER,
    product_id VARCHAR REFERENCES products(product_id),
    seller_id VARCHAR,
    price DECIMAL(10, 2),
    freight_value DECIMAL(10, 2)
);

CREATE TABLE order_reviews (
    review_id VARCHAR PRIMARY KEY,
    order_id VARCHAR REFERENCES orders(order_id),
    review_score TINYINT,
    review_comment_title VARCHAR,
    review_comment_message VARCHAR,
    review_creation_date TIMESTAMP,
    embedding FLOAT[768]
);

-- Create HNSW Index for Plan B Vector-First Pushdown
CREATE INDEX idx_reviews_embedding_hnsw
ON order_reviews USING HNSW (embedding)
WITH (metric = 'cosine', ef_construction = 128, M = 16);

-- Create Full-Text BM25 Index for Hybrid Lexical + Dense RRF
PRAGMA create_fts_index(
    'order_reviews',
    'review_id',
    'review_comment_title',
    'review_comment_message',
    stemmer = 'english',
    stopwords = 'english',
    lower = 1
);
```

---

### 2. Commercial "Bring-Your-Own-Export" (BYOE) Adapter Schema

To deploy `DualQuery` for real local businesses (e.g., B2B supply chain/textile dispatch & defect tracking or e-commerce sellers) without requiring a custom database rewrite, `DualQuery` includes a universal 2-table ingestion adapter (`dualquery ingest --transactions tx.csv --feedback notes.csv`):

| Canonical Table | Mapped Columns | Real-World Commercial Example |
| :--- | :--- | :--- |
| `canonical_transactions` | `tx_id`, `party_id`, `category`, `region`, `amount`, `freight_or_tax`, `status`, `tx_timestamp`, `delay_days` | Invoices, dispatch challans, Shopify/WooCommerce orders, or ERP ledger rows. |
| `canonical_feedback` | `note_id`, `tx_id`, `severity_score`, `raw_text`, `note_timestamp`, `embedding FLOAT[768]` | Customer return notes, quality rejection remarks, WhatsApp complaints, or support tickets. |

---

### 3. Production Repository Structure

```text
dualquery/
├── pyproject.toml                     # Entrypoint for `pip install dualquery-mcp`
├── README.md                          # Architecture, benchmarks, and 60-sec quickstart
├── docs/
│   ├── 00_EXECUTIVE_SUMMARY_AND_VISION.md
│   ├── 01_SYSTEM_ARCHITECTURE_AND_OPTIMIZER_SPEC.md
│   ├── 02_ZERO_TRUST_SECURITY_AND_MCP_PROTOCOL.md
│   ├── 03_COMPETITIVE_TEARDOWN_AND_STAFF_DEFENSE.md
│   └── 04_DATA_SCHEMAS_BENCHMARKS_AND_ROADMAP.md
├── src/
│   └── dualquery/
│       ├── __init__.py
│       ├── cli.py                     # CLI entrypoint ('dualquery ingest', 'dualquery audit', 'dualquery serve')
│       ├── config.py                  # Settings, RBAC role manifests, cost thresholds
│       ├── ingestion/
│       │   ├── olist_loader.py        # Builds olist.duckdb + FastEmbed vectors + FTS
│       │   ├── byoe_adapter.py        # Universal CSV/Excel commercial audit ingester
│       │   └── literal_indexer.py     # Builds categorical enum lookup dictionary
│       ├── security/
│       │   ├── ast_guard.py           # 5-gate SQLGlot AST validator, RBAC policies & RLS injector
│       │   └── pii_vault.py           # HMAC key tokenizer & regex/Luhn text scrubber
│       ├── engine/
│       │   ├── literal_grounder.py    # Pre-SQL fuzzy + semantic value resolver
│       │   ├── cardinality_probe.py   # <2ms COUNT(*) selectivity estimator
│       │   ├── physical_planner.py    # Switches between Plan A (SIMD) & Plan B (HNSW+BM25 RRF)
│       │   ├── executor.py            # DuckDB query runner & telemetry collector
│       │   └── audit_runner.py        # Commercial BYOE Financial Leakage Engine & Terminal Dashboard
│       └── mcp/
│           └── server.py              # FastMCP Zero-Trust server (stdio & SSE/HTTP)
├── tests/
│   ├── test_byoe_audit.py             # Fuzzy header mapping & commercial audit tests
│   ├── test_mcp_server.py             # FastMCP tools & RBAC tests
│   └── test_security_compiler.py      # Automated 30 red-team security gate tests
└── benchmarks/
    ├── golden_50_hybrid_queries.json  # Ground-truth SQL + Vector evaluation suite (50 queries)
    ├── evaluate_optimizer.py          # Benchmarks Plan A vs Plan B recall & latency
    └── evaluate_security.py           # Red-team prompt injection & PII leak test suite
```

---

### 4. 5-Phase Execution & Launch Plan

1. **Phase 1: Unified DuckDB Engine & Ingestion Pipeline (Days 1–3)**
   * Ingest the 100k Olist relational dataset and English translated reviews into a single `olist.duckdb` file.
   * Compute 768-d embeddings locally using `FastEmbed` (`BAAI/bge-base-en-v1.5`) and build both the `vss` HNSW index and `fts` BM25 index.
   * Build the `literal_indexer.py` lookup table over distinct categories, states, and payment types.

2. **Phase 2: Cost-Based Physical Planner & RRF Engine (Days 4–6)**
   * Implement `cardinality_probe.py` and `physical_planner.py` switching at $N_{\text{cand}} = 5,000$ (AVX2-calibrated) between **Plan A (Exact SIMD Cosine + BM25 on filtered rows)** and **Plan B (HNSW + BM25 RRF $\rightarrow$ Relational Hash Join)**.
   * Run `benchmarks/evaluate_optimizer.py` across all 50 golden queries to generate empirical latency and recall curves.

3. **Phase 3: `SQLGlot` AST Security Compiler & Local PII Vault (Days 7–8)**
   * Implement the 5-gate `SQLGlot` AST rewriter and HMAC-SHA256 PII tokenizer (`pii_vault.py`).
   * Pass a 30-query red-team adversarial test suite (`DROP TABLE`, unauthorized `JOIN` to restricted tables, CTE smuggling, PII extraction attempts) with a 100% block rate (tested via both pytest and benchmark runner).

4. **Phase 4: Zero-Trust MCP Server Packaging & Registry Publication (Days 9–11)**
   * Wrap the engine in `FastMCP` (`src/dualquery/mcp/server.py`), configure `pyproject.toml`, and publish `dualquery-mcp` to PyPI.
   * Test live inside Claude Desktop and Cursor, then submit to **Smithery.ai, Glama.ai, and Awesome-MCP-Servers**.

5. **Phase 5: Commercial BYOE Audit Case Study & Visual Telemetry UI (Days 12–14)**
   * Run the `byoe_adapter.py` pipeline on a real commercial dataset (order/dispatch invoices + customer return/quality complaint records) to produce a verified financial leakage case study alongside the Bloomberg-style terminal telemetry dashboard.
