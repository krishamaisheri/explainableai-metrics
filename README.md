<p align="center">
  <h1 align="center">🔍 Operationalising Explainability &amp; Transparency<br/>in a Public-Sector RAG System</h1>
  <p align="center">
    A metric-based governance layer that evaluates the <strong>explainability</strong> and <strong>transparency</strong><br/>
    of a Retrieval-Augmented Generation (RAG) system deployed for UK public-sector policy queries.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?logo=flask&logoColor=white" alt="Flask" />
  <img src="https://img.shields.io/badge/LLMs-LLaMA%20%7C%20Mistral-FF6F00" alt="LLMs" />
  <img src="https://img.shields.io/badge/OpenRouter-API-6366F1" alt="OpenRouter" />
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License" />
  <img src="https://img.shields.io/badge/Tests-Mocked%20LLM-blue" alt="Tests" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [The 8 Explainability Metrics](#-the-8-explainability-metrics)
  - [M1 — Input Attribution Consistency Score (IACS)](#m1--input-attribution-consistency-score-iacs)
  - [M2 — Input Contradiction Rate (ICR)](#m2--input-contradiction-rate-icr)
  - [M3 — Internal Reasoning Consistency Score (IRCS)](#m3--internal-reasoning-consistency-score-ircs)
  - [M4 — Explanation–Decision Alignment Score (EDAS)](#m4--explanationdecision-alignment-score-edas)
  - [M5 — Structured Explanation Completeness Score (SECS)](#m5--structured-explanation-completeness-score-secs)
  - [M6 — Policy Grounding Similarity Score (PGSS)](#m6--policy-grounding-similarity-score-pgss)
  - [M7 — Explanation Stability Index (ESI)](#m7--explanation-stability-index-esi)
  - [M8 — Explanation Density Ratio (EDR)](#m8--explanation-density-ratio-edr)
- [Metric Aggregation & Alert Engine](#-metric-aggregation--alert-engine)
- [RAG Pipeline](#-rag-pipeline)
- [Data Ingestion Pipeline](#-data-ingestion-pipeline)
- [Web Interfaces](#-web-interfaces)
  - [Monitoring Dashboard](#monitoring-dashboard)
  - [Chatbot Interface](#chatbot-interface)
- [Batch Evaluation](#-batch-evaluation)
- [Production Monitoring](#-production-monitoring)
- [LLM Observability](#-llm-observability)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Testing](#-testing)
- [Technology Stack](#-technology-stack)
- [License](#-license)

---

## 🧠 Overview

This project implements a **quantitative governance layer** for evaluating the explainability of AI-generated responses in a UK public-sector context. Citizens query the system about benefit eligibility (Universal Credit, Housing Benefit, Disability Living Allowance, etc.), and the RAG pipeline produces policy-grounded explanations. Each response is then evaluated across **8 independent explainability metrics**, each capturing a different dimension of transparency, faithfulness, and structural quality.

The system is designed to:

- **Quantify explainability** using formal, reproducible metrics with mathematical definitions.
- **Detect governance violations** via configurable RAG (Red-Amber-Green) alert thresholds.
- **Provide full auditability** through per-metric computation traces, LLM call logging, and JSONL audit trails.
- **Support real-time and batch evaluation** with both interactive web interfaces and a CLI batch runner.

### Key Capabilities

| Capability                   | Description                                                                                  |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| **8 Explainability Metrics** | Each with a formal mathematical definition, LLM-as-a-judge evaluation, and computation trace |
| **Weighted Aggregation**     | Configurable weights combining metrics into a single Explainability Score                    |
| **RAG Pipeline**             | ChromaDB-backed retrieval with cosine similarity over UK government policy documents         |
| **Alert Engine**             | Red/Amber/Green thresholds with automatic escalation logging                                 |
| **Monitoring Dashboard**     | Flask-based dashboard with Chart.js visualisations and rolling averages                      |
| **Chatbot Interface**        | Standalone web chatbot for interactive query evaluation                                      |
| **Batch Runner**             | 10-query evaluation suite with structured CLI output and summary tables                      |
| **Full Observability**       | Per-metric LLM call tracking, computation traces, and JSONL audit logs                       |

---

## 🏗 System Architecture

```
                           ┌────────────────────┐
                           │   Citizen Query     │
                           └─────────┬──────────┘
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │         RAG PIPELINE             │
                    │                                  │
                    │  ┌───────────┐  ┌─────────────┐  │
                    │  │  Vector   │→ │  ChromaDB    │  │
                    │  │  Store    │  │  (JSON+NumPy)│  │
                    │  └───────────┘  └──────┬──────┘  │
                    │       cosine similarity │         │
                    │                        ▼         │
                    │  ┌──────────────────────────┐    │
                    │  │  Top-K Policy Chunks     │    │
                    │  └────────────┬─────────────┘    │
                    │               │                   │
                    │               ▼                   │
                    │  ┌──────────────────────────┐    │
                    │  │  LLM Generator           │    │
                    │  │  (LLaMA / Mistral via    │    │
                    │  │   OpenRouter)             │    │
                    │  └────────────┬─────────────┘    │
                    └───────────────┼──────────────────┘
                                    │
                                    ▼
                     ┌──────────────────────────────┐
                     │   EXPLAINABILITY EVALUATION   │
                     │          LAYER                │
                     │                               │
                     │  ┌─────┐ ┌─────┐ ┌──────┐    │
                     │  │IACS │ │ ICR │ │ IRCS │    │
                     │  └─────┘ └─────┘ └──────┘    │
                     │  ┌─────┐ ┌─────┐ ┌──────┐    │
                     │  │EDAS │ │SECS │ │ PGSS │    │
                     │  └─────┘ └─────┘ └──────┘    │
                     │  ┌─────┐ ┌─────┐              │
                     │  │ ESI │ │ EDR │              │
                     │  └─────┘ └─────┘              │
                     └──────────────┬───────────────┘
                                    │
                                    ▼
                    ┌─────────────────────────────────┐
                    │    METRIC AGGREGATOR             │
                    │  Weighted sum → Explainability   │
                    │  Score + RAG Alert Detection     │
                    └──────────────┬──────────────────┘
                                   │
                        ┌──────────┴──────────┐
                        ▼                     ▼
              ┌─────────────────┐   ┌────────────────────┐
              │  JSONL Monitor  │   │   Web Dashboard /  │
              │  scores.jsonl   │   │   Chatbot / CLI    │
              │  traces.jsonl   │   │   pretty_print     │
              │  escalations    │   └────────────────────┘
              └─────────────────┘
```

### Data Flow

1. **Query Ingestion** — A citizen submits a policy question (e.g., _"Am I eligible for council housing? I am 25 with two children."_)
2. **Retrieval** — The vector store performs cosine similarity search against embedded UK government policy chunks, returning the top-5 most relevant passages.
3. **Generation** — The LLM (LLaMA 3.1 8B or Mistral 7B via OpenRouter) produces a structured explanation grounded in retrieved policy context.
4. **Metric Evaluation** — All 8 explainability metrics are computed independently, each using LLM-as-a-judge or embedding-based analysis.
5. **Aggregation** — Weighted sum produces a single Explainability Score; alert thresholds flag violations.
6. **Logging** — Results, traces, and escalations are persisted to JSONL files.
7. **Presentation** — Results are displayed via the monitoring dashboard, chatbot, or colourised CLI output.

---

## 📐 The 8 Explainability Metrics

Each metric is a self-contained Python module in `metrics/` exposing a `compute(query, explanation, **kwargs) → float` function that returns a score in **[0, 1]**.

All metrics emit **computation traces** via the `MetricTraceCollector`, capturing intermediate values, formula steps, and the final score for full auditability.

---

### M1 — Input Attribution Consistency Score (IACS)

**File:** [`metrics/iacs.py`](metrics/iacs.py)  
**Purpose:** Measures whether the explanation correctly reflects and uses relevant attributes from the user query.

**Formula:**

```
IACS = |A_E ∩ A_Q| / |A_Q|
```

| Symbol      | Meaning                                                                              |
| ----------- | ------------------------------------------------------------------------------------ |
| `A_Q`       | Set of factual attributes extracted from the **query** (e.g., age, income, children) |
| `A_E`       | Set of factual attributes extracted from the **explanation**                         |
| `A_E ∩ A_Q` | Intersection — attributes from the query that appear in the explanation              |

**How it works:**

1. LLM extracts attributes from the query → `A_Q`
2. LLM extracts attributes from the explanation → `A_E`
3. Score = overlap / total query attributes _(1.0 = all query attributes referenced)_

**Edge Case:** If no attributes found in query → returns `1.0` (trivially consistent).

**Weight:** `0.25` | **Alert Thresholds:** Green ≥ `0.95`, Amber ≥ `0.90`

---

### M2 — Input Contradiction Rate (ICR)

**File:** [`metrics/icr.py`](metrics/icr.py)  
**Purpose:** Detects whether the explanation contradicts factual statements made in the user query.

**Formula:**

```
ICR = 1 − (contradictions / total_facts)
```

| Symbol           | Meaning                                                      |
| ---------------- | ------------------------------------------------------------ |
| `contradictions` | Number of user-stated facts that the explanation contradicts |
| `total_facts`    | Total number of factual statements extracted from the query  |

**How it works:**

1. LLM extracts every factual statement from the query.
2. For each fact, LLM checks whether the explanation contradicts it (binary `true`/`false` judgement).
3. Score = 1 minus the contradiction ratio _(1.0 = no contradictions)_.

**Edge Case:** If no facts extracted → returns `1.0`.

**Weight:** _Not in weighted aggregate_ (used as an independent diagnostic)

---

### M3 — Internal Reasoning Consistency Score (IRCS)

**File:** [`metrics/ircs.py`](metrics/ircs.py)  
**Purpose:** Checks whether reasoning statements within the explanation contradict each other.

**Formula:**

```
IRCS = 1 − T / P
where T = contradictory pairs, P = k(k−1)/2
```

| Symbol | Meaning                                        |
| ------ | ---------------------------------------------- |
| `k`    | Number of reasoning clauses in the explanation |
| `P`    | Total pairwise combinations: `k × (k−1) / 2`   |
| `T`    | Number of contradictory pairs found            |

**How it works:**

1. LLM segments the explanation into individual reasoning clauses.
2. For every pair of clauses `(i, j)`, LLM checks for mutual contradiction.
3. Score = 1 minus the ratio of contradictory pairs _(1.0 = internally consistent)_.

**Edge Case:** If fewer than 2 clauses → returns `1.0` (can't contradict).

**Weight:** `0.25` | **Alert Thresholds:** Green ≥ `0.97`, Amber ≥ `0.93`

---

### M4 — Explanation–Decision Alignment Score (EDAS)

**File:** [`metrics/edas.py`](metrics/edas.py)  
**Purpose:** Measures whether the reasoning in the explanation logically entails the final decision.

**Formula:**

```
EDAS = P(Decision | Reasoning)   (entailment probability)
```

**How it works:**

1. LLM separates the explanation into **reasoning** (the justification) and **decision** (the final conclusion).
2. LLM rates the entailment strength on a `[0.0, 1.0]` scale.
3. Score is clamped to `[0, 1]` _(1.0 = perfect alignment)_.

**Edge Case:** If reasoning or decision cannot be extracted → returns `0.0`.

**Weight:** `0.10` | **Alert Thresholds:** Green ≥ `0.93`, Amber ≥ `0.88`

---

### M5 — Structured Explanation Completeness Score (SECS)

**File:** [`metrics/secs.py`](metrics/secs.py)  
**Purpose:** Detects the presence of four required structural components in the explanation.

**Formula:**

```
SECS = (# components present) / 4
```

**Required Components:**

| #   | Component             | Description                                         |
| --- | --------------------- | --------------------------------------------------- |
| 1   | `user_factors`        | References to the user's specific circumstances     |
| 2   | `policy_rule`         | Citation or paraphrase of the relevant policy/rule  |
| 3   | `logical_application` | Logical application of the rule to the user's case  |
| 4   | `decision_link`       | Clear link between reasoning and the final decision |

**How it works:**

1. LLM analyses the explanation and returns a `true`/`false` for each of the four components.
2. Score = count of present components / 4 _(1.0 = all components present)_.

**Weight:** `0.15` | **Alert Thresholds:** Green ≥ `0.90`, Amber ≥ `0.85`

---

### M6 — Policy Grounding Similarity Score (PGSS)

**File:** [`metrics/pgss.py`](metrics/pgss.py)  
**Purpose:** Measures semantic support of each explanation clause by the policy corpus using embedding cosine similarity.

**Formula:**

```
PGSS = (# supported clauses) / (total clauses)

A clause is "supported" if:
    max(cosine_similarity(clause_embedding, policy_embedding_i)) ≥ threshold
```

**How it works:**

1. LLM segments the explanation into individual factual clauses.
2. Each clause is embedded using `sentence-transformers/all-MiniLM-L6-v2`.
3. For each clause, the maximum cosine similarity against all policy document embeddings is computed.
4. A clause is "supported" if `max_sim ≥ 0.7` (configurable via `PGSS_SIMILARITY_THRESHOLD`).
5. Score = fraction of supported clauses _(1.0 = fully policy-grounded)_.

**Edge Cases:**

- No policy texts → returns `0.0`
- No clauses extracted → returns `1.0`

**Weight:** `0.10` | **Alert Thresholds:** Green ≥ `0.85`, Amber ≥ `0.75`

---

### M7 — Explanation Stability Index (ESI)

**File:** [`metrics/esi.py`](metrics/esi.py)  
**Purpose:** Measures variance of explanation across repeated generation runs.

**Formula:**

```
ESI = 1 − average_cosine_distance(repeated_outputs)

cosine_distance(a, b) = 1 − cosine_similarity(a, b)
```

**How it works:**

1. The LLM generates `N` explanations for the same query (default `N=3`, configurable via `ESI_REPEAT_RUNS`).
2. All explanations are embedded using `all-MiniLM-L6-v2`.
3. Pairwise cosine distances are computed for all `N(N−1)/2` pairs.
4. Score = 1 minus the mean distance _(1.0 = perfectly stable, identical outputs)_.

**Edge Case:** Only 1 explanation → returns `1.0`.

**Weight:** `0.10` | **Alert Thresholds:** Green ≥ `0.90`, Amber ≥ `0.80`

---

### M8 — Explanation Density Ratio (EDR)

**File:** [`metrics/edr.py`](metrics/edr.py)  
**Purpose:** Measures the proportion of explanation tokens devoted to substantive reasoning versus filler/boilerplate.

**Formula:**

```
EDR = reasoning_tokens / (reasoning_tokens + filler_tokens)
```

| Category             | Examples                                                                       |
| -------------------- | ------------------------------------------------------------------------------ |
| **Reasoning tokens** | Logical arguments, policy references, factual analysis                         |
| **Filler tokens**    | Greetings, hedging language, boilerplate disclaimers, pleasantries, repetition |

**How it works:**

1. LLM classifies the explanation's token (word) count into `reasoning_tokens` and `filler_tokens`.
2. Score = reasoning ratio _(1.0 = entirely substantive)_.

**Edge Case:** Total tokens = 0 → returns `0.0`.

**Weight:** `0.05` | **Alert Thresholds:** Green ≥ `0.60`, Amber ≥ `0.50`

---

## 📊 Metric Aggregation & Alert Engine

**File:** [`aggregator.py`](aggregator.py)

The aggregator computes a **weighted Explainability Score** from all metrics and applies configurable RAG thresholds.

### Weighted Aggregation Formula

```
Explainability Score = Σ (weight_i × score_i)   for all metrics with assigned weights
```

### Current Metric Weights

| Metric   | Weight | Governance Focus                     |
| -------- | ------ | ------------------------------------ |
| **IACS** | `0.25` | Input attribution faithfulness       |
| **IRCS** | `0.25` | Internal reasoning consistency       |
| **SECS** | `0.15` | Structural completeness              |
| **EDAS** | `0.10` | Decision alignment                   |
| **PGSS** | `0.10` | Policy grounding                     |
| **ESI**  | `0.10` | Output stability                     |
| **EDR**  | `0.05` | Reasoning density                    |
| **ICR**  | —      | Diagnostic only (independent signal) |

> **Note:** Weights are configured in `config.py` → `METRIC_WEIGHTS` and must sum to `1.0`.

### RAG Alert Thresholds

The alert engine uses a **three-tier system** (Red / Amber / Green) for each metric:

| Metric        | 🟢 Green (≥) | 🟡 Amber (≥) | 🔴 Red (<) |
| ------------- | ------------ | ------------ | ---------- |
| IACS          | `0.95`       | `0.90`       | `0.90`     |
| IRCS          | `0.97`       | `0.93`       | `0.93`     |
| EDAS          | `0.93`       | `0.88`       | `0.88`     |
| SECS          | `0.90`       | `0.85`       | `0.85`     |
| PGSS          | `0.85`       | `0.75`       | `0.75`     |
| ESI           | `0.90`       | `0.80`       | `0.80`     |
| EDR           | `0.60`       | `0.50`       | `0.50`     |
| **Aggregate** | `0.94`       | `0.90`       | `0.90`     |

When a metric's score falls below its **green** threshold, an alert is raised with severity `Amber` or `Red`.

---

## 🔗 RAG Pipeline

**Files:** [`rag_pipeline.py`](rag_pipeline.py) → [`vector_store.py`](vector_store.py)

### Retrieval

```python
query → vector_store.query(query, top_k=5) → top-5 policy chunks ranked by cosine similarity
```

The vector store is a **custom JSON+NumPy implementation** (not ChromaDB directly) that stores:

- **Documents:** Policy chunks with metadata (`text`, `source`, `category`, `page`, `chunk_index`) in `chroma_db/documents.json`
- **Embeddings:** Pre-computed `all-MiniLM-L6-v2` embeddings in `chroma_db/embeddings.npy`

At query time, the query is embedded, cosine similarity is computed against all stored embeddings, and the top-K chunks are returned.

### Generation

The retrieved chunks are formatted into a structured prompt instructing the LLM to answer with:

1. Relevant user factors
2. Applicable policy rule(s)
3. How the rule applies to the user's case
4. Decision / recommendation

The LLM generation uses `meta-llama/llama-3.1-8b-instruct:free` via OpenRouter.

---

## 📄 Data Ingestion Pipeline

**Files:** [`build_vector_store.py`](build_vector_store.py) → [`pdf_extractor.py`](pdf_extractor.py) → [`vector_store.py`](vector_store.py)

### Pipeline Steps

```
data/*.pdf  →  PDF Extraction (PyMuPDF + OCR)  →  extracted/*.md  →  Chunking  →  Embedding  →  chroma_db/
```

1. **PDF Extraction** (`pdf_extractor.py`):
   - Phase 1: Standard text layer extraction via PyMuPDF.
   - Phase 2: Native OCR at 300 DPI as fallback for scanned documents.
   - Output saved as `.md` files in `extracted/<category>/`.
   - Intelligent caching: skips re-extraction if `.md` already exists.

2. **Chunking** (configurable):
   - `CHUNK_SIZE = 500` characters
   - `CHUNK_OVERLAP = 100` characters
   - Sliding window with overlap for context preservation.

3. **Embedding & Ingestion** (`vector_store.py`):
   - Encodes all chunks using `sentence-transformers/all-MiniLM-L6-v2`.
   - Persists documents as JSON + embeddings as `.npy`.

### Policy Data Categories

```
data/
├── universal-credit/        # 10 documents
├── housing-benefits/        # 6 documents
├── financial-help-disabled/ # 9 documents
├── general/                 # 12 documents
└── dla-disability-living-allowance-benefit/  # 3 documents
```

---

## 🌐 Web Interfaces

### Monitoring Dashboard

**Files:** [`dashboard.py`](dashboard.py) → [`templates/dashboard.html`](templates/dashboard.html)  
**URL:** `http://localhost:8000`

A Flask-powered governance dashboard providing:

- **Real-time Metric Cards** — Displays all 8 metric scores with colour-coded status.
- **Rolling Average Charts** — Chart.js line charts showing metric trends over the last 20 evaluations.
- **Alert Timeline** — Chronological view of threshold breaches.
- **Escalation Log** — Flagged interactions requiring human review.
- **Live Evaluation** — `/api/evaluate` endpoint for on-demand query scoring.

**API Endpoints:**

| Method | Endpoint                  | Description                         |
| ------ | ------------------------- | ----------------------------------- |
| `GET`  | `/`                       | Render the dashboard HTML           |
| `GET`  | `/api/scores?limit=50`    | Return recent score log entries     |
| `GET`  | `/api/averages?window=20` | Return rolling averages             |
| `GET`  | `/api/escalations`        | Return escalation log entries       |
| `POST` | `/api/evaluate`           | Evaluate a query `{"query": "..."}` |

---

### Chatbot Interface

**Files:** [`chatbot.py`](chatbot.py) → [`templates/chatbot.html`](templates/chatbot.html)  
**URL:** `http://localhost:8001`

A standalone Flask chatbot interface for interactive query evaluation. Send a citizen-style policy question and receive:

- The LLM-generated explanation.
- All 8 metric scores with visual breakdowns.
- Computation traces showing mathematical steps.
- LLM call logs with prompts, responses, and timings.

**API Endpoints:**

| Method | Endpoint     | Description                         |
| ------ | ------------ | ----------------------------------- |
| `GET`  | `/`          | Serve the chatbot HTML interface    |
| `POST` | `/api/query` | Evaluate a query `{"query": "..."}` |

> **Note:** This is a fully standalone module — deleting it does not affect the pipeline or any other component.

---

## 🚀 Batch Evaluation

**File:** [`batch_runner.py`](batch_runner.py)

Runs 10 predefined UK policy queries through the full pipeline with structured CLI output:

| #   | Query Category                           | Example                                              |
| --- | ---------------------------------------- | ---------------------------------------------------- |
| Q1  | Universal Credit (low income + children) | _"I am unemployed, have two children..."_            |
| Q2  | Universal Credit (higher income)         | _"I work full-time and earn £3,200/month..."_        |
| Q3  | Housing Benefit (appeal)                 | _"My Housing Benefit application was rejected..."_   |
| Q4  | Housing Benefit (eligibility)            | _"I rent privately and receive Pension Credit..."_   |
| Q5  | Disability support (general)             | _"I have a long-term disability..."_                 |
| Q6  | DLA (child case)                         | _"My 10-year-old child has mobility needs..."_       |
| Q7  | DLA (adult confusion)                    | _"I am 30 years old with a physical disability..."_  |
| Q8  | UC + fluctuating income                  | _"I am self-employed, income varies £400–£1,200..."_ |
| Q9  | Housing Benefit + employment             | _"I work part-time and rent privately..."_           |
| Q10 | Multi-policy overlap                     | _"I am unemployed, disabled, and rent privately..."_ |

```bash
python batch_runner.py
```

Produces per-query results with colourised progress bars, plus a summary table with aggregate scores, LLM call counts, timings, and failure counts.

---

## 📈 Production Monitoring

**File:** [`monitor.py`](monitor.py)

Three JSONL log files provide a full audit trail:

| Log File                 | Contents                                                              | Purpose                       |
| ------------------------ | --------------------------------------------------------------------- | ----------------------------- |
| `logs/scores.jsonl`      | Timestamp, query preview, all metric scores, aggregate score, alerts  | High-level score history      |
| `logs/traces.jsonl`      | Full query, explanation, per-metric computation traces, LLM call logs | Detailed audit/debug trail    |
| `logs/escalations.jsonl` | Timestamp, aggregate score, alerts, query preview, status             | Governance escalation records |

**Features:**

- **Rolling Averages:** Computes sliding-window averages over the last `N` evaluations for trend detection.
- **Automatic Escalation:** When any alert is triggered, an escalation record is automatically written.
- **Thin Score Log:** The score log is kept lightweight for dashboard performance; the trace log holds the full detail.

---

## 🔭 LLM Observability

**File:** [`llm_client.py`](llm_client.py)

Every LLM call is tracked through two singleton collectors:

### LLMTracker

Records per-call data for every LLM invocation:

| Field      | Description                                                               |
| ---------- | ------------------------------------------------------------------------- |
| `model`    | OpenRouter model identifier used                                          |
| `duration` | Wall-clock time in seconds                                                |
| `caller`   | Which metric/component made the call (e.g., `"IACS"`, `"RAG_GENERATION"`) |
| `success`  | Whether the call succeeded                                                |
| `prompt`   | First 500 characters of the prompt                                        |
| `response` | First 1000 characters of the response                                     |

Provides per-caller stats (`total_calls`, `successful`, `failed`, `total_time`, `models_used`) and a global summary.

### MetricTraceCollector

Captures the **computation trace** from each metric, including:

- The mathematical formula used
- Intermediate values (e.g., extracted attributes, clause pairs, similarity scores)
- Step-by-step computation breakdown
- Final score

### JSON Parsing Robustness

The LLM client includes production-grade JSON parsing:

- **Bracket-depth matching** to extract JSON objects from conversational or markdown-fenced responses.
- **Control character sanitisation** to handle literal newlines/tabs inside JSON strings.
- **Retry with exponential backoff** (3 attempts by default).

---

## 📁 Project Structure

```
explainableai-metrics/
│
├── config.py                    # Central configuration (API keys, weights, thresholds)
├── llm_client.py                # OpenRouter LLM wrapper + LLMTracker + MetricTraceCollector
├── pipeline.py                  # End-to-end orchestrator: Query → RAG → Metrics → Aggregate → Log
├── aggregator.py                # Weighted metric aggregation + RAG alert engine
├── monitor.py                   # JSONL logging, rolling averages, escalation detection
├── pretty_print.py              # ANSI-coloured CLI output with progress bars
│
├── metrics/                     # 8 explainability metric modules
│   ├── __init__.py              # METRIC_REGISTRY — maps names to compute functions
│   ├── iacs.py                  # M1: Input Attribution Consistency Score
│   ├── icr.py                   # M2: Input Contradiction Rate
│   ├── ircs.py                  # M3: Internal Reasoning Consistency Score
│   ├── edas.py                  # M4: Explanation–Decision Alignment Score
│   ├── secs.py                  # M5: Structured Explanation Completeness Score
│   ├── pgss.py                  # M6: Policy Grounding Similarity Score
│   ├── esi.py                   # M7: Explanation Stability Index
│   └── edr.py                   # M8: Explanation Density Ratio
│
├── rag_pipeline.py              # RAG: retrieve top-k chunks → generate explanation
├── vector_store.py              # JSON+NumPy vector store (cosine similarity retrieval)
├── build_vector_store.py        # One-time CLI: extract PDFs → chunk → embed → store
├── pdf_extractor.py             # PDF text extraction (PyMuPDF + OCR fallback at 300 DPI)
│
├── dashboard.py                 # Flask monitoring dashboard (port 8000)
├── chatbot.py                   # Flask chatbot interface (port 8001)
├── templates/
│   ├── dashboard.html           # Dashboard frontend (Chart.js, metric cards, alerts)
│   └── chatbot.html             # Chatbot frontend (interactive query UI)
│
├── batch_runner.py              # Batch evaluation of 10 test queries
│
├── data/                        # UK government policy PDFs (5 categories, 40 documents)
│   ├── universal-credit/
│   ├── housing-benefits/
│   ├── financial-help-disabled/
│   ├── general/
│   └── dla-disability-living-allowance-benefit/
│
├── sample_policies/             # Lightweight .txt policy files for quick testing
│   ├── benefits_universal_credit.txt
│   ├── council_tax_reduction.txt
│   └── housing_eligibility.txt
│
├── extracted/                   # Auto-generated .md files from PDF extraction (gitignored)
├── chroma_db/                   # Persisted vector store (gitignored)
├── logs/                        # JSONL audit logs (gitignored)
│
├── tests/                       # Pytest suite — all LLM calls mocked
│   ├── __init__.py
│   ├── test_metrics.py          # 19 unit tests for all 8 metrics
│   ├── test_aggregator.py       # Aggregation + alert detection tests
│   └── test_pipeline.py         # End-to-end pipeline integration tests
│
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (gitignored)
└── .gitignore
```

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11+**
- An [OpenRouter](https://openrouter.ai) API key (free tier available for LLaMA and Mistral models)

### 1. Clone & Install

```bash
git clone https://github.com/krishamaisheri/explainableai-metrics.git
cd explainableai-metrics
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Create your .env file
echo "OPENROUTER_API_KEY=sk-or-your-key-here" > .env
```

<details>
<summary><b>📋 Full .env Options</b></summary>

```env
# Required
OPENROUTER_API_KEY=sk-or-your-key-here

# Optional — Model overrides
REASONING_MODEL=meta-llama/llama-3.1-8b-instruct:free
NLI_MODEL=mistralai/mistral-7b-instruct:free
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Optional — Operational
LOG_DIR=logs
ESI_REPEAT_RUNS=3
PGSS_SIMILARITY_THRESHOLD=0.7
```

</details>

### 3. Build the Vector Store (one-time)

```bash
python build_vector_store.py
```

This extracts text from all PDFs in `data/`, chunks them, computes embeddings, and persists the vector store to `chroma_db/`.

### 4. Run a Single Evaluation

```bash
python pipeline.py "Am I eligible for council housing? I am 25 with two children."
```

### 5. Launch the Monitoring Dashboard

```bash
python dashboard.py
# Open http://localhost:8000
```

### 6. Launch the Chatbot

```bash
python chatbot.py
# Open http://localhost:8001
```

### 7. Run Batch Evaluation

```bash
python batch_runner.py
```

### 8. Run Tests (no API key needed)

```bash
python -m pytest tests/ -v
```

---

## ⚙ Configuration

All configuration is centralised in [`config.py`](config.py) and loaded from environment variables via `.env`.

### Models

| Setting           | Default                                  | Purpose                                      |
| ----------------- | ---------------------------------------- | -------------------------------------------- |
| `REASONING_MODEL` | `meta-llama/llama-3.1-8b-instruct:free`  | Main generation model                        |
| `NLI_MODEL`       | `mistralai/mistral-7b-instruct:free`     | Metric evaluation (NLI / classification)     |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Sentence embeddings for PGSS, ESI, retrieval |

### Operational Parameters

| Setting                     | Default | Purpose                                                      |
| --------------------------- | ------- | ------------------------------------------------------------ |
| `LOG_DIR`                   | `logs`  | Directory for JSONL audit logs                               |
| `ESI_REPEAT_RUNS`           | `3`     | Number of repeated generations for ESI stability measurement |
| `PGSS_SIMILARITY_THRESHOLD` | `0.7`   | Cosine similarity threshold for PGSS policy grounding        |

---

## 🧪 Testing

The test suite uses **fully mocked LLM calls** — no API key or network access required.

```bash
# Run all tests with verbose output
python -m pytest tests/ -v

# Run only metric tests
python -m pytest tests/test_metrics.py -v

# Run only aggregator tests
python -m pytest tests/test_aggregator.py -v

# Run only pipeline tests
python -m pytest tests/test_pipeline.py -v
```

### Test Coverage

| Test File            | Tests                                                                    | Coverage                                               |
| -------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------ |
| `test_metrics.py`    | 19 tests                                                                 | All 8 metrics: edge cases, partial scores, full scores |
| `test_aggregator.py` | Aggregation correctness, alert triggering, threshold boundary conditions |
| `test_pipeline.py`   | End-to-end pipeline with mocked RAG + metrics                            |

**Test Design Philosophy:**

- Every metric is tested for **full overlap**, **partial overlap**, and **edge cases** (empty input, missing data).
- LLM responses are mocked using `unittest.mock.patch` to ensure deterministic, fast tests.
- Embedding models are mocked with fixed NumPy arrays for PGSS and ESI tests.

---

## 🛠 Technology Stack

| Category           | Technology                               | Purpose                                     |
| ------------------ | ---------------------------------------- | ------------------------------------------- |
| **Language**       | Python 3.11+                             | Core implementation                         |
| **LLMs**           | LLaMA 3.1 8B / Mistral 7B                | Reasoning, classification, generation       |
| **LLM Gateway**    | [OpenRouter](https://openrouter.ai)      | Unified API for multiple model providers    |
| **LLM SDK**        | `openai>=1.0.0`                          | OpenAI-compatible client for OpenRouter     |
| **Embeddings**     | `sentence-transformers/all-MiniLM-L6-v2` | Semantic similarity & retrieval             |
| **Vector Store**   | Custom JSON + NumPy                      | Persistent document & embedding storage     |
| **PDF Extraction** | PyMuPDF (`fitz`)                         | Text extraction + native OCR at 300 DPI     |
| **Web Framework**  | Flask 3.0+                               | Dashboard and chatbot servers               |
| **Charts**         | Chart.js                                 | Rolling average visualisations              |
| **CLI Output**     | Colorama + ANSI codes                    | Rich terminal formatting with progress bars |
| **Testing**        | pytest + unittest.mock                   | Fully mocked LLM test suite                 |
| **Config**         | python-dotenv                            | Environment variable management             |

---

## 📜 License

MIT
