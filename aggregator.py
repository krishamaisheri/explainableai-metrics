"""
Metric Aggregator.

Computes the weighted Explainability Score from all 8 metrics
and applies alert thresholds to flag low-scoring interactions.
Tracks per-metric timing, failures, traces, and LLM call details.
"""

import time
import logging
import config
from metrics import METRIC_REGISTRY
from llm_client import tracker, trace_collector

logger = logging.getLogger(__name__)


def aggregate(scores: dict[str, float]) -> dict:
    """
    Aggregate individual metric scores into a single Explainability Score.

    Parameters
    ----------
    scores : dict[str, float]
        Mapping from metric name (e.g. "IACS") to score ∈ [0, 1].

    Returns
    -------
    dict
        {
            "metric_scores": {…},
            "aggregate_score": float,
            "alerts": [{…}, …]
        }
    """
    weights = config.METRIC_WEIGHTS
    weighted_sum = sum(
        weights.get(name, 0.0) * scores.get(name, 0.0)
        for name in weights
    )

    # ── Alert detection ──────────────────────────────────────
    alerts = []
    thresholds = config.ALERT_THRESHOLDS

    for metric_name, levels in thresholds.items():
        score = weighted_sum if metric_name == "Aggregate" else scores.get(metric_name, 0.0)

        if score < levels["green"]:
            severity = "Red" if score < levels["amber"] else "Amber"
            alerts.append({
                "metric": metric_name,
                "score": round(score, 4),
                "threshold_green": levels["green"],
                "threshold_amber": levels["amber"],
                "severity": severity
            })

    return {
        "metric_scores": {k: round(v, 4) for k, v in scores.items()},
        "aggregate_score": round(weighted_sum, 4),
        "alerts": alerts,
    }


def compute_all_metrics(
    query: str,
    explanation: str,
    **extra_kwargs,
) -> dict:
    """
    Compute every registered metric and return the aggregated result.

    Returns enriched data including per-metric timing, failures,
    LLM call statistics, computation traces, and per-metric LLM call logs.
    """
    # Reset tracker and trace collector for this evaluation run
    tracker.reset()
    trace_collector.reset()

    scores = {}
    metric_timings = {}
    metric_failures = {}

    for name, compute_fn in METRIC_REGISTRY.items():
        t0 = time.time()
        try:
            scores[name] = compute_fn(query, explanation, **extra_kwargs)
            metric_timings[name] = round(time.time() - t0, 3)
        except Exception as exc:
            scores[name] = 0.0
            metric_timings[name] = round(time.time() - t0, 3)
            metric_failures[name] = str(exc)
            logger.error("Metric %s failed: %s", name, exc)

    result = aggregate(scores)
    result["metric_timings"] = metric_timings
    result["metric_failures"] = metric_failures
    result["llm_stats"] = tracker.get_stats()
    result["llm_global"] = tracker.get_global_summary()

    # ── Collect per-metric traces and LLM call logs ──────────
    result["metric_traces"] = trace_collector.get_all_traces()

    metric_llm_calls = {}
    for name in METRIC_REGISTRY:
        calls = tracker.get_calls_for(name)
        if calls:
            metric_llm_calls[name] = calls
    # Also capture RAG_GENERATION calls
    rag_calls = tracker.get_calls_for("RAG_GENERATION")
    if rag_calls:
        metric_llm_calls["RAG_GENERATION"] = rag_calls

    result["metric_llm_calls"] = metric_llm_calls

    return result
