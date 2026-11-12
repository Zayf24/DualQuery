# DualQuery (`dualquery-mcp`)
## Document 00: Executive Summary & Vision (v2.0)

### Project Identity

**`DualQuery` (`dualquery-mcp`)** is an open-source, Zero-Trust Hybrid Relational-Semantic Query Engine and Model Context Protocol (MCP) Server. It compiles natural language business questions into cost-optimized execution plans that join structured SQL tables with unstructured customer text (reviews, support tickets, operational memos) inside a single embedded engine—enforcing AST-level access control and local PII redaction so raw customer records never enter an external LLM's context window.

---

### 1. The Core Problem Statement

In 95% of real-world enterprises, operational and financial truth is physically split across two incompatible storage paradigms:

* **70% Structured Relational Data (OLTP / OLAP Databases):** Hundreds of thousands of order records, payment settlements, item prices, shipping freight charges, and delivery timestamps.
* **30% Unstructured Natural Language:** Customer review comments, support tickets, return notes, and internal operational memos.

When an operator, financial analyst, or product leader asks a compound business question:

> *"Find all delivered orders over R$300 paid via credit card where customers complained about damaged packaging or missing parts, calculate the total refund liability, and rank the worst-performing seller states."*

#### Why Every Existing Approach Breaks in Production:

1. **Pure Vector RAG Fails:** Vector indexes cannot execute deterministic mathematical aggregations (`SUM`, `AVG`, `COUNT`, `GROUP BY`), compute timestamp deltas, or perform multi-table foreign-key joins.
2. **Pure Text-to-SQL Fails:** Relational queries rely on rigid `LIKE '%missing%'` string matching, completely missing semantic equivalents such as *"half the screws were absent from the box"* or *"arrived shattered"*.
3. **1M-Context Window Stuffing Fails:** Dumping 50,000 raw database rows into frontier long-context models costs **$1.50–$3.00 per query**, introduces **25–40 seconds Time-To-First-Token (TTFT)**, suffers from LLM arithmetic hallucinations over large tables, and violates data privacy compliance by streaming raw customer PII to external model providers.
4. **Standard Database MCP Servers Leak Confidential Data:** Naive MCP servers (`server-postgres`, `server-sqlite`) execute raw `SELECT` queries and dump unredacted customer rows straight back to the cloud LLM host.

---

### 2. Dual-Track Product & Real-User Adoption Strategy

To operate as a real-world production tool with verifiable adoption and commercial impact rather than a static dataset demo, `DualQuery` ships in three synchronized layers:

| Track | Target User | Deliverable & Mechanism | Verifiable Metric |
| :--- | :--- | :--- | :--- |
| **Track 1: Open-Source SDK & Zero-Trust MCP Server** | AI Engineers, Data Teams, Cursor / Claude Desktop users | `pip install dualquery-mcp` — plugs into any local DuckDB or PostgreSQL database via `stdio` or `Streamable HTTP`, exposing hybrid SQL + semantic joins with local PII scrubbing. | PyPI downloads, GitHub stars, listings on Smithery / Glama / PulseMCP registries. |
| **Track 2: Commercial "Bring-Your-Own-Export" (BYOE) Audit Engine** | D2C Brands, Local B2B Suppliers, Operations Managers | Zero-integration CLI/Web diagnostic (`dualquery audit --orders orders.csv --notes returns.csv`) that maps order/invoice ledgers against customer return/complaint logs to quantify financial leakage by defect cluster. | Real commercial case study over live invoice + return/dispatch records quantifying actual ₹/$ revenue leakage. |
| **Track 3: Reproducible Public Benchmark Suite** | Technical Interviewers & CI Pipeline | Automated evaluation harness over **100,000 real Brazilian E-Commerce (Olist) orders** across 9 relational tables + 100,000 customer reviews. | Execution Accuracy ($EX$), Vector Recall@K, and P95 Engine Latency (`<35ms`). |

---

### 3. Verified Benchmark Summary

* **Optimizer Plan Accuracy (`benchmarks/evaluate_optimizer.py`):** **100.0%** plan match across 50 golden hybrid queries (`Plan A` P50: `315ms`, P95: `409ms`; `Plan B` P50: `1,003ms`, P95: `1,073ms` including local query embedding + OLAP aggregation).
* **Security Gate Block Rate (`benchmarks/evaluate_security.py`):** **100.0%** (`30/30` adversarial SQL injection, schema-escape, and unauthorized column-qualifier attacks blocked across Gates 1–5).
