# DualQuery (`dualquery-mcp`)
## Document 03: Competitive Teardown & Staff Defense (v2.0)

### 1. Comprehensive 2026 Competitive Matrix

| Platform / Framework | Valuation / Status | Core Strength | Why It Fails the `DualQuery` Use Case |
| :--- | :--- | :--- | :--- |
| **Glean** | **$7.2B Valuation**, >$100M ARR | 100+ SaaS connectors (Slack, Drive, Jira) with continuous ACL crawling. | Blind to relational data warehouses. Cannot run multi-table SQL joins or exact mathematical aggregations over transactional tables. |
| **Onyx (formerly Danswer)** | **$10M Seed** (Khosla, First Round, YC W24) | Self-hostable hybrid document search (BM25 + dense vectors). | Built for document chunk retrieval; has no cost-based SQL-to-vector join planner or semantic metrics layer. |
| **Snowflake Cortex (Analyst + Search)** | Public Cloud Data Warehouse Add-on | Native Text-to-SQL over semantic views + hybrid search inside Snowflake. | **Hard vendor lock-in & cloud-only cost.** Requires all data to live inside Snowflake credits; cannot run locally over DuckDB/PostgreSQL or act as a zero-egress local MCP server. |
| **LOTUS (`lotus-ai`)** | UC Berkeley / Stanford OSS Research | Declarative semantic operators (`sem_join`, `sem_filter`, `sem_agg`) in Pandas. | Executes semantic operators by invoking LLM calls across rows (high token cost and latency) rather than pushing down into unified DuckDB SQL + local SIMD/HNSW + BM25 RRF with AST RBAC guardrails. |
| **Hebbia / Dust.tt** | **$700M** (Hebbia) / **$40M Series B** (Dust) | Multi-document matrix extraction (Hebbia) and modular team agents (Dust). | Designed for asynchronous document synthesis; cannot execute sub-50ms local relational joins with deterministic financial aggregation. |

---

### 2. The Staff Engineer Grilling Room (Updated v2.0 Defenses)

#### Q1: "You claim `<40ms` latency, yet your system uses an LLM for query planning and synthesis. Isn't that benchmark misleading?"
> **Your Answer:**  
> *"We strictly separate **Engine Execution Latency** from **End-to-End Agent Turn Latency** in our telemetry. Once the MCP server receives the structured hybrid plan, the pre-flight cardinality probe, DuckDB relational filter, SIMD/HNSW vector search, BM25 Reciprocal Rank Fusion, and OLAP aggregation execute locally in **11ms to 38ms (P95)**.  
> The full end-to-end turn—including LLM intent decomposition and final answer synthesis—takes **~1.1 seconds**, which is still a **22x latency reduction** and **98% cost reduction** compared to stuffing 50,000 JSON rows into a 1M-context frontier model (which takes 25–40 seconds and hallucinates arithmetic)."*

---

#### Q2: "Why did you drop ChromaDB and consolidate everything into DuckDB?"
> **Your Answer:**  
> *"In v1, running relational filters in DuckDB and passing candidate IDs to ChromaDB created a severe serialization bottleneck: a 4% selectivity filter on 1M rows meant serializing 40,000 UUID strings across Python into ChromaDB's SQLite metadata filter, blowing past SQLite variable limits and spiking latency to 600ms+.  
> By consolidating relational tables, `FLOAT[384]` embeddings (`vss` HNSW), and BM25 lexical indexes (`fts`) into a single DuckDB engine, candidate keys never leave C++ memory. More importantly, when our pre-flight cardinality probe sees $N_{\text{cand}} \le 1,500$, DuckDB runs an exact SIMD `array_cosine_distance` scan directly on the filtered subset in `<4ms`, achieving **100% vector recall** and completely avoiding the HNSW pre-filtering graph disconnection problem."*

---

#### Q3: "DuckDB's `vss` HNSW index has known persistence and lateral join constraints. How did you engineer around them?"
> **Your Answer:**  
> *"Two specific ways:  
> First, DuckDB's HNSW index is in-memory by default and requires `SET hnsw_enable_experimental_persistence = true` for disk persistence; to guarantee crash safety in production, embeddings are stored canonically in the native `FLOAT[384]` column, and our startup health check verifies or rebuilds the HNSW index in `<1.8s` if a WAL recovery invalidated it.  
> Second, because DuckDB's lateral `vss_join` macro falls back to a quadratic brute-force scan rather than using the HNSW index, our physical optimizer never uses `vss_join`. Instead, in Plan B it executes a top-K `ORDER BY array_cosine_distance(...) LIMIT K` query (which triggers the HNSW index) inside a CTE and hash-joins that bounded CTE against the relational tables."*

---

#### Q4: "How do you prevent Text-to-SQL from generating valid SQL that returns 0 rows because of mismatched category strings?"
> **Your Answer:**  
> *"That is the classic 'dirty string literal' failure mode. Before the agent compiles SQL, it invokes our `resolve_entity_literals` tool, which runs a sub-5ms trigram + embedding match against a pre-indexed dictionary of distinct categorical column values (`product_category_name`, `customer_state`, `payment_type`). That guarantees the SQL compiler writes `WHERE product_category_name = 'beleza_saude'` instead of guessing `'health_and_beauty'`."*
