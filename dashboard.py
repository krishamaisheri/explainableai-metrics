"""
Monitoring Dashboard — Flask application.

Serves a single-page governance dashboard with:
  - Real-time metric cards
  - Rolling-average charts (Chart.js)
  - Alert timeline
  - Escalation log
  - /evaluate API endpoint for live scoring
"""

import json
import logging
from flask import Flask, render_template, request, jsonify

import monitor
import config

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Render the monitoring dashboard."""
    scores = monitor.read_score_log(limit=50)
    escalations = monitor.read_escalation_log(limit=20)
    averages = monitor.rolling_averages(window=20)
    thresholds = config.ALERT_THRESHOLDS
    weights = config.METRIC_WEIGHTS
    return render_template(
        "dashboard.html",
        scores_json=json.dumps(scores),
        escalations_json=json.dumps(escalations),
        averages_json=json.dumps(averages),
        thresholds_json=json.dumps(thresholds),
        weights_json=json.dumps(weights),
    )


@app.route("/api/scores")
def api_scores():
    """Return recent scores as JSON."""
    limit = request.args.get("limit", 50, type=int)
    return jsonify(monitor.read_score_log(limit=limit))


@app.route("/api/averages")
def api_averages():
    """Return rolling averages."""
    window = request.args.get("window", 20, type=int)
    return jsonify(monitor.rolling_averages(window=window))


@app.route("/api/escalations")
def api_escalations():
    """Return escalation log."""
    return jsonify(monitor.read_escalation_log(limit=50))


@app.route("/api/evaluate", methods=["POST"])
def api_evaluate():
    """
    Evaluate a query through the full pipeline.

    POST JSON: {"query": "Am I eligible for council housing?"}
    """
    data = request.get_json(force=True)
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Missing 'query' field"}), 400

    # Import here to avoid circular import at module level
    from pipeline import evaluate
    result = evaluate(query)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=8000, host="0.0.0.0")
