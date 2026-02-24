"""
Production Monitor.

* Stores per-interaction scores to a JSON-lines log.
* Computes rolling averages.
* Detects threshold breaches and writes escalation records.
"""

import json
import os
import datetime
import config

_SCORE_LOG = os.path.join(config.LOG_DIR, "scores.jsonl")
_ESCALATION_LOG = os.path.join(config.LOG_DIR, "escalations.jsonl")


def _ensure_dir():
    os.makedirs(config.LOG_DIR, exist_ok=True)


def log_interaction(result: dict, query: str = "", explanation: str = "") -> None:
    """Append a scored interaction to the score log."""
    _ensure_dir()
    entry = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "query_preview": query[:120],
        "metric_scores": result.get("metric_scores", {}),
        "aggregate_score": result.get("aggregate_score", 0.0),
        "alerts": result.get("alerts", []),
    }
    with open(_SCORE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Write escalation record if alerts were triggered
    if entry["alerts"]:
        escalation = {
            "timestamp": entry["timestamp"],
            "aggregate_score": entry["aggregate_score"],
            "alerts": entry["alerts"],
            "query_preview": entry["query_preview"],
            "status": "flagged",
        }
        with open(_ESCALATION_LOG, "a") as f:
            f.write(json.dumps(escalation) + "\n")


def read_score_log(limit: int = 200) -> list[dict]:
    """Return the most recent *limit* score-log entries."""
    _ensure_dir()
    if not os.path.exists(_SCORE_LOG):
        return []
    with open(_SCORE_LOG) as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    return entries[-limit:]


def read_escalation_log(limit: int = 50) -> list[dict]:
    """Return the most recent *limit* escalation entries."""
    _ensure_dir()
    if not os.path.exists(_ESCALATION_LOG):
        return []
    with open(_ESCALATION_LOG) as f:
        lines = f.readlines()
    entries = [json.loads(l) for l in lines if l.strip()]
    return entries[-limit:]


def rolling_averages(window: int = 20) -> dict[str, float]:
    """Compute rolling averages of metric scores over the last *window* entries."""
    entries = read_score_log(limit=window)
    if not entries:
        return {}
    metric_names = list(entries[0].get("metric_scores", {}).keys())
    averages = {}
    for name in metric_names:
        vals = [e["metric_scores"].get(name, 0.0) for e in entries]
        averages[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    agg_vals = [e.get("aggregate_score", 0.0) for e in entries]
    averages["Aggregate"] = round(sum(agg_vals) / len(agg_vals), 4) if agg_vals else 0.0
    return averages
