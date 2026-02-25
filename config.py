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
# ── Metric weights (must sum to 1.0) ───────────────────────────
METRIC_WEIGHTS = {
    "IACS": 0.25,
    "IRCS": 0.25,
    "EDAS": 0.10,
    "SECS": 0.15,
    "PGSS": 0.10,
    "ESI":  0.10,
    "EDR":  0.05,
}

# ── Alert thresholds ───────────────────────────────────────────
# Format: { MetricName: { "green": float, "amber": float } }
# Red is anything below "amber". Amber is [amber, green). Green is >= green.
ALERT_THRESHOLDS = {
    "IACS":      {"green": 0.95, "amber": 0.90},
    "IRCS":      {"green": 0.97, "amber": 0.93},
    "EDAS":      {"green": 0.93, "amber": 0.88},
    "SECS":      {"green": 0.90, "amber": 0.85},
    "PGSS":      {"green": 0.85, "amber": 0.75},
    "ESI":       {"green": 0.90, "amber": 0.80},
    "EDR":       {"green": 0.60, "amber": 0.50},
    "Aggregate": {"green": 0.94, "amber": 0.90},
}

# ── Operational parameters ─────────────────────────────────────
LOG_DIR = os.getenv("LOG_DIR", "logs")
ESI_REPEAT_RUNS = int(os.getenv("ESI_REPEAT_RUNS", 3))
PGSS_SIMILARITY_THRESHOLD = float(os.getenv("PGSS_SIMILARITY_THRESHOLD", 0.7))
