"""
Integration test — full pipeline with mocked LLM.
"""

from unittest.mock import patch, MagicMock
import numpy as np


@patch("metrics.esi._get_embedding_model")
@patch("metrics.pgss._get_embedding_model")
@patch("metrics.pgss.llm_client.call_llm_json")
@patch("metrics.edr.llm_client.call_llm_json")
@patch("metrics.secs.llm_client.call_llm_json")
@patch("metrics.edas.llm_client.call_llm_json")
@patch("metrics.ircs.llm_client.call_llm_json")
@patch("metrics.icr.llm_client.call_llm_json")
@patch("metrics.iacs.llm_client.call_llm_json")
def test_full_pipeline(
    mock_iacs, mock_icr, mock_ircs, mock_edas, mock_secs,
    mock_edr, mock_pgss_json, mock_pgss_model, mock_esi_model,
):
    # IACS: full overlap
    mock_iacs.side_effect = [
        {"attributes": ["age"]},
        {"attributes": ["age"]},
    ]
    # ICR: no contradictions
    mock_icr.side_effect = [
        {"facts": ["fact1"]},
        {"contradicts": False},
    ]
    # IRCS: no contradictions
    mock_ircs.side_effect = [
        {"clauses": ["A", "B"]},
        {"contradicts": False},
    ]
    # EDAS: good entailment
    mock_edas.side_effect = [
        {"reasoning": "R", "decision": "D"},
        {"entailment_score": 0.9},
    ]
    # SECS: all present
    mock_secs.return_value = {
        "user_factors": True, "policy_rule": True,
        "logical_application": True, "decision_link": True,
    }
    # EDR: mostly reasoning
    mock_edr.return_value = {"reasoning_tokens": 40, "filler_tokens": 10}

    # PGSS: good similarity
    mock_pgss_json.return_value = {"clauses": ["clause1"]}
    pgss_encoder = MagicMock()
    pgss_encoder.encode.side_effect = [
        np.array([[1.0, 0.0]]),
        np.array([[0.98, 0.2]]),
    ]
    mock_pgss_model.return_value = pgss_encoder

    # ESI: stable
    esi_encoder = MagicMock()
    esi_encoder.encode.return_value = np.array([[1.0, 0.0], [0.99, 0.1]])
    mock_esi_model.return_value = esi_encoder

    from aggregator import compute_all_metrics

    result = compute_all_metrics(
        query="Am I eligible?",
        explanation="Based on your age of 25, the policy states...",
        policy_texts=["Policy rule 1"],
        precomputed_explanations=["exp1", "exp2"],
    )

    assert "metric_scores" in result
    assert "aggregate_score" in result
    assert 0 <= result["aggregate_score"] <= 1
    assert len(result["metric_scores"]) == 8
    for name in ["IACS", "ICR", "IRCS", "EDAS", "SECS", "PGSS", "ESI", "EDR"]:
        assert name in result["metric_scores"]
        assert 0 <= result["metric_scores"][name] <= 1
