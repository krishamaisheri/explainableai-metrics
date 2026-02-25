"""
Standalone Chatbot Interface.

A completely independent Flask app that provides a web-based
chatbot for querying the explainability pipeline.

Usage:
    python chatbot.py

Runs on http://localhost:8001

NOTE: This is standalone — deleting this file won't affect
      the pipeline or any other module.
"""

import json
import logging
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
logging.basicConfig(level=logging.WARNING)


@app.route("/")
def index():
    """Serve the chatbot HTML page."""
    return render_template("chatbot.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """
    Evaluate a query through the full pipeline.

    POST JSON: {"query": "Am I eligible for council housing?"}
    Returns the full pipeline result as JSON.
    """
    data = request.get_json(force=True)
    query = data.get("query", "").strip()

    if not query:
        return jsonify({"error": "Please enter a query."}), 400

    try:
        from pipeline import evaluate
        result = evaluate(query, silent=True)
        return jsonify(result)
    except Exception as exc:
        logging.error("Pipeline error: %s", exc)
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print("\n🤖 Chatbot running at http://localhost:8001\n")
    app.run(debug=False, port=8001, host="0.0.0.0")
