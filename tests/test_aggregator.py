"""
Tests for the metric aggregator.
"""

import pytest
from aggregator import aggregate


class TestAggregate:
    def test_perfect_scores(self):
        scores = {
            "IACS": 1.0, "ICR": 1.0, "IRCS": 1.0, "EDAS": 1.0,
            "SECS": 1.0, "PGSS": 1.0, "ESI": 1.0, "EDR": 1.0,
        }
        result = aggregate(scores)
        assert result["aggregate_score"] == 1.0
        assert result["alerts"] == []

    def test_low_ircs_triggers_alert(self):
        scores = {
            "IACS": 1.0, "ICR": 1.0, "IRCS": 0.80, "EDAS": 1.0,
            "SECS": 1.0, "PGSS": 1.0, "ESI": 1.0, "EDR": 1.0,
        }
        result = aggregate(scores)
        alert_metrics = [a["metric"] for a in result["alerts"]]
        assert "IRCS" in alert_metrics

    def test_low_aggregate_triggers_alert(self):
        scores = {
            "IACS": 0.5, "ICR": 0.5, "IRCS": 0.5, "EDAS": 0.5,
            "SECS": 0.5, "PGSS": 0.5, "ESI": 0.5, "EDR": 0.5,
        }
        result = aggregate(scores)
        assert result["aggregate_score"] == 0.5
        alert_metrics = [a["metric"] for a in result["alerts"]]
        assert "Aggregate" in alert_metrics

    def test_weights_sum_correctly(self):
        """Weighted sum with known values."""
        scores = {
            "IACS": 0.9, "ICR": 0.8, "IRCS": 0.95, "EDAS": 0.85,
            "SECS": 0.75, "PGSS": 0.7, "ESI": 0.9, "EDR": 0.8,
        }
        result = aggregate(scores)
        expected = (
            0.20 * 0.9 + 0.15 * 0.8 + 0.20 * 0.95 + 0.15 * 0.85 +
            0.10 * 0.75 + 0.10 * 0.7 + 0.05 * 0.9 + 0.05 * 0.8
        )
        assert abs(result["aggregate_score"] - round(expected, 4)) < 1e-4
