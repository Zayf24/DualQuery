# DualQuery: Ideation Evolution & Scrutiny Log
## Full Research Trajectory, User Feedback, and Subagent Stress-Tests

> **Context:** This document preserves the exact chronological sequence of questions, critical critiques, subagent evaluations, and architectural refinements that led to the creation of `DualQuery`.

---

## 📅 Round 1: Initial Problem Statement & Candidate Scrutiny

### User Request:
> *"Let's decide the title for this project. I want to make this as a product so I can talk to the interviewer about real-world application and real-world figures. Do a deep research and provide 2–3 unique, mainstream topics."*

### Candidates Explored:
1. `SRE-Pulse` (Incident Post-Mortems & Runbook Retrieval)
2. `Cross-Repo Microservice GraphRAG` (Distributed schema drift & blast-radius analysis)
3. `ColPali Layout-Aware Multimodal RAG` (Vision-native document intelligence over complex PDFs)
4. `Acoustic-Grounded Speech RAG` (Vocal prosody & timestamped audio retrieval)

### User's Critical Counter-Questions:
> 1. *"Where will we get the real data for these?"*
> 2. *"An interviewer can argue: 'Why not just use Claude Code or Codex CLI with a prompt in the terminal to do this? Everyone has Pro plans and agentic coding tools now.' We cannot use outdated models; we must evaluate against late-2026 frontier models (GPT-6 Astra, Claude Fable, Gemini Carbon, DeepSeek V4) with 1M+ context."*
> 3. *"Summon subagents acting as senior developers to ruthlessly scrutinize each idea and ask counter-questions based on real-world data."*

---

## 🔍 Round 2: Subagent Stress-Test (Senior Staff AI Architect)

### 1. `SRE-Pulse` $\rightarrow$ 🛑 VERDICT: SCRAP
* **The Fatal Flaw:** Static markdown runbooks are trivially grepped by Claude Code (`rg` + terminal) in 3 seconds inside the engineer's terminal.
* **The Data Impossibility:** Real P0 telemetry (Datadog metrics, Jaeger traces, Kubernetes logs) is strictly confidential enterprise IP protected by SOC-2. Public post-mortems (like `danluu/post-mortems`) are high-level retrospective blog posts, not raw telemetry. Synthetic fake logs are immediately spotted by senior hiring managers as a toy tutorial project.

### 2. User Feedback on Developer Tools, Vision & Audio:
> * *"One crucial thing to remember is that no one writes code by hand now, everyone uses agentic coding tools like Cursor, Claude Code, Codex, Antigravity IDE. For cross-repo, tools like Graphify create a 'brain' containing full codebase knowledge that Claude Code uses for development. So cross-repo code tools are already saturated."*
> * *"ColPali is mostly an OCR/PDF extraction project rather than a core RAG product project."*
> * *"Audio RAG feels irrelevant / not what I want to build."*

---

## 💡 Round 3: The Enterprise Product Pivot

### User Challenge:
> *"Find an authentic, production-grade enterprise product RAG system that real businesses use and pay for, which is NOT a coding assistant, NOT an OCR project, and NOT an audio project. Use subagents to stress-test."*

### Subagent Principal Architect Evaluation:
We evaluated three high-value enterprise domains:
1. `ChronosRAG`: Bi-temporal regulatory & legal compliance (valid time vs assertion time, handling supersession DAGs).
2. `ZeroTrust-RAG`: Authorization-gated enterprise search (the Glean/Onyx architecture, eliminating document leakage and retrieval starvation using Roaring Bitmaps).
3. `DualQuery / HybridRelational-RAG`: Unified Text-to-SQL + Vector Join Engine for enterprise data warehouses.

### Why `DualQuery` Emerged as Rank #1:
* **The Universal Enterprise Reality:** 70% of enterprise data lives in relational SQL tables (revenue, orders, payments, logistics, churn); 30% lives in unstructured text (reviews, support tickets, call notes).
* **The Failure of Existing Tools:**
  * Naive RAG cannot do math (`SUM`, `AVG`, `GROUP BY`) or foreign-key joins.
  * Pure Text-to-SQL cannot understand semantic text nuance.
  * 1M-context frontier LLMs hallucinate calculations when given large row sets, cost $1.50–$3.00/query, and violate customer PII data compliance.
* **100% Free Public Data:** The Brazilian E-Commerce Dataset by Olist (100,000 real orders across 9 relational tables + 100,000 real customer reviews).

---

## ❓ Round 4: Agentic RAG & Cross-Company Scope

### Question 1: *"What is Agentic RAG, and does DualQuery fall into Agentic RAG?"*
* **Answer:** **YES, DualQuery is the textbook definition of production Agentic RAG.**
  * *Traditional RAG* is a static one-way conveyor belt: `Prompt -> Embed -> Top-K -> LLM`. It has zero brain and zero self-correction.
  * *Agentic RAG* is an autonomous goal-driven system with:
    1. **Dynamic Routing & Planning:** Deconstructs complex queries into structured SQL and semantic vector goals.
    2. **Multi-Tool Calling:** Orchestrates between a relational database engine (DuckDB) and a vector similarity engine (ChromaDB).
    3. **Cost-Based Query Optimization:** Chooses between Plan A (SQL-First Pushdown) and Plan B (Vector-First Pushdown) based on table statistics.
    4. **Reflection & Self-Correction:** Adjusts similarity thresholds or retries if queries return 0 results.

### Question 2: *"Can we frame it as a central RAG system for the whole company spanning Tech, Finance, VPs, PMs, Board Members, Customer Care, Marketing?"*
* **The Brutal Reality Check:** Pitching a *"universal search box for everyone"* is the classic 2023 naive RAG anti-pattern. A single vector index breaks across different departments (code needs AST, finance needs SQL tables, customer support needs sub-500ms ticket lookups).
* **The Winning Reframe:**  
  **"A Federated Enterprise Agentic Hub with Role-Governed Intent Routing and Hybrid Relational-Semantic Execution."**  
  * Bridges the multi-billion-dollar blind spot of Glean and Microsoft 365 Copilot (who only index text and are blind to relational data warehouses).
  * Demonstrates three distinct roles using real Olist data:
    * *Finance:* Pure SQL over orders/payments $\to$ 0% math hallucination.
    * *Customer Support:* Semantic vector search over 100k reviews $\to$ ticket resolution.
    * *Executive / VP:* Hybrid multi-hop join (SQL orders + vector reviews + executive memos).
