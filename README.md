# Operationalising Explainability & Transparency in a Public-Sector RAG System

A metric-based governance layer evaluating the explainability and transparency of a Retrieval-Augmented Generation (RAG) system used by a UK public-sector organisation.

## Architecture

```
Query → Retriever → Policy Docs → Generator (LLaMA/Mistral via OpenRouter)
                                        │
                                        ▼
                              Explainability Evaluation Layer
                              ├── IACS  – Input Attribution Consistency
                              ├── ICR   – Input Contradiction Rate
                              ├── IRCS  – Internal Reasoning Consistency
                              ├── EDAS  – Explanation–Decision Alignment
                              ├── SECS  – Structured Explanation Completeness
                              ├── PGSS  – Policy Grounding Similarity
                              ├── ESI   – Explanation Stability Index
                              └── EDR   – Explanation Density Ratio
                                        │
                                        ▼
                              Metric Aggregator → Monitoring Dashboard
```

## Quick Start

```bash
# 1. Clone and install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your OpenRouter API key

# 3. Run tests (no API key needed — all LLM calls mocked)
python -m pytest tests/ -v

# 4. Launch dashboard
python dashboard.py
# Open http://localhost:5000

# 5. Run a single evaluation (needs API key)
python pipeline.py "Am I eligible for council housing? I am 25 with two children."
```

## Project Structure

```
├── config.py              # Central configuration
├── llm_client.py          # OpenRouter LLM wrapper
├── metrics/               # 8 explainability metrics
│   ├── iacs.py            # Input Attribution Consistency Score
│   ├── icr.py             # Input Contradiction Rate
│   ├── ircs.py            # Internal Reasoning Consistency Score
│   ├── edas.py            # Explanation–Decision Alignment Score
│   ├── secs.py            # Structured Explanation Completeness
│   ├── pgss.py            # Policy Grounding Similarity Score
│   ├── esi.py             # Explanation Stability Index
│   └── edr.py             # Explanation Density Ratio
├── aggregator.py          # Weighted aggregation + alert engine
├── monitor.py             # JSONL logging + rolling averages
├── rag_pipeline.py        # Minimal RAG (retrieve + generate)
├── pipeline.py            # End-to-end orchestrator
├── dashboard.py           # Flask monitoring dashboard
├── templates/dashboard.html
├── sample_policies/       # Demo UK policy documents
└── tests/                 # Pytest suite (mocked LLM)
```

## Metric Weights

| Metric | Weight | Alert Threshold |
|--------|--------|----------------|
| IACS   | 0.20   | < 0.90         |
| ICR    | 0.15   | —              |
| IRCS   | 0.20   | < 0.95         |
| EDAS   | 0.15   | < 0.90         |
| SECS   | 0.10   | —              |
| PGSS   | 0.10   | —              |
| ESI    | 0.05   | —              |
| EDR    | 0.05   | —              |
| **Aggregate** | — | **< 0.92** |

## Technology

- **LLMs**: LLaMA / Mistral via [OpenRouter](https://openrouter.ai)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Dashboard**: Flask + Chart.js
- **Tests**: pytest with mocked LLM calls

## License

MIT
