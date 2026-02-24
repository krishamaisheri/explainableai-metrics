"""
Central configuration module.
Loads all settings from environment variables / .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── OpenRouter API ──────────────────────────────────────────────
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ── Model identifiers ──────────────────────────────────────────
REASONING_MODEL = os.getenv("REASONING_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
NLI_MODEL = os.getenv("NLI_MODEL", "mistralai/mistral-7b-instruct:free")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ── Metric weights (must sum to 1.0) ───────────────────────────
METRIC_WEIGHTS = {
    "IACS": float(os.getenv("WEIGHT_IACS", 0.20)),
    "ICR":  float(os.getenv("WEIGHT_ICR",  0.15)),
    "IRCS": float(os.getenv("WEIGHT_IRCS", 0.20)),
    "EDAS": float(os.getenv("WEIGHT_EDAS", 0.15)),
    "SECS": float(os.getenv("WEIGHT_SECS", 0.10)),
    "PGSS": float(os.getenv("WEIGHT_PGSS", 0.10)),
    "ESI":  float(os.getenv("WEIGHT_ESI",  0.05)),
    "EDR":  float(os.getenv("WEIGHT_EDR",  0.05)),
}

# ── Alert thresholds ───────────────────────────────────────────
ALERT_THRESHOLDS = {
    "IRCS":      float(os.getenv("ALERT_IRCS",      0.95)),
    "IACS":      float(os.getenv("ALERT_IACS",      0.90)),
    "EDAS":      float(os.getenv("ALERT_EDAS",      0.90)),
    "Aggregate": float(os.getenv("ALERT_AGGREGATE", 0.92)),
}

# ── Operational parameters ─────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "logs")
ESI_REPEAT_RUNS = int(os.getenv("ESI_REPEAT_RUNS", 3))
PGSS_SIMILARITY_THRESHOLD = float(os.getenv("PGSS_SIMILARITY_THRESHOLD", 0.7))
