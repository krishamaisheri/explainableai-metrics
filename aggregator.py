"""
Metric Aggregator.

Computes the weighted Explainability Score from all 8 metrics
and applies alert thresholds to flag low-scoring interactions.
"""

import config
from metrics import METRIC_REGISTRY


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
            "alerts": [{"metric": …, "score": …, "threshold": …}, …]
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

    for metric_name, threshold in thresholds.items():
        if metric_name == "Aggregate":
            if weighted_sum < threshold:
                alerts.append({
                    "metric": "Aggregate",
                    "score": round(weighted_sum, 4),
                    "threshold": threshold,
                })
        else:
            score = scores.get(metric_name, 0.0)
            if score < threshold:
                alerts.append({
                    "metric": metric_name,
                    "score": round(score, 4),
                    "threshold": threshold,
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

    Extra keyword arguments (e.g. policy_texts) are forwarded to
    individual metric compute functions.
    """
    scores = {}
    for name, compute_fn in METRIC_REGISTRY.items():
        try:
            scores[name] = compute_fn(query, explanation, **extra_kwargs)
        except Exception as exc:
            scores[name] = 0.0
            import logging
            logging.getLogger(__name__).error(
                "Metric %s failed: %s", name, exc
            )
    return aggregate(scores)
