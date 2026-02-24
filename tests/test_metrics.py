"""
Unit tests for all 8 explainability metrics.

All LLM calls are mocked — no API key required.
"""

import pytest
from unittest.mock import patch, MagicMock
import json


# ── Helpers ──────────────────────────────────────────────────
def _mock_llm_json(return_value: dict):
    """Create a side_effect for call_llm_json that returns *return_value*."""
    return MagicMock(return_value=return_value)


# ═══════════════════════════════════════════════════════════════
# 1. IACS
# ═══════════════════════════════════════════════════════════════
class TestIACS:
    @patch("metrics.iacs.llm_client.call_llm_json")
    def test_full_overlap(self, mock_json):
        """All query attributes referenced in explanation → 1.0"""
        mock_json.side_effect = [
            {"attributes": ["age", "location", "children"]},       # query
            {"attributes": ["age", "location", "children", "income"]},  # explanation
        ]
        from metrics.iacs import compute
        assert compute("q", "e") == 1.0

    @patch("metrics.iacs.llm_client.call_llm_json")
    def test_partial_overlap(self, mock_json):
        mock_json.side_effect = [
            {"attributes": ["age", "location", "children"]},
            {"attributes": ["age"]},
        ]
        from metrics.iacs import compute
        assert abs(compute("q", "e") - 1 / 3) < 1e-6

    @patch("metrics.iacs.llm_client.call_llm_json")
    def test_no_query_attributes(self, mock_json):
        mock_json.side_effect = [
            {"attributes": []},
            {"attributes": ["age"]},
        ]
        from metrics.iacs import compute
        assert compute("q", "e") == 1.0  # trivially consistent


# ═══════════════════════════════════════════════════════════════
# 2. ICR
# ═══════════════════════════════════════════════════════════════
class TestICR:
    @patch("metrics.icr.llm_client.call_llm_json")
    def test_no_contradictions(self, mock_json):
        mock_json.side_effect = [
            {"facts": ["fact1", "fact2"]},
            {"contradicts": False},
            {"contradicts": False},
        ]
        from metrics.icr import compute
        assert compute("q", "e") == 1.0

    @patch("metrics.icr.llm_client.call_llm_json")
    def test_one_contradiction(self, mock_json):
        mock_json.side_effect = [
            {"facts": ["fact1", "fact2"]},
            {"contradicts": True},
            {"contradicts": False},
        ]
        from metrics.icr import compute
        assert compute("q", "e") == 0.5

    @patch("metrics.icr.llm_client.call_llm_json")
    def test_no_facts(self, mock_json):
        mock_json.return_value = {"facts": []}
        from metrics.icr import compute
        assert compute("q", "e") == 1.0


# ═══════════════════════════════════════════════════════════════
# 3. IRCS
# ═══════════════════════════════════════════════════════════════
class TestIRCS:
    @patch("metrics.ircs.llm_client.call_llm_json")
    def test_no_contradictions(self, mock_json):
        mock_json.side_effect = [
            {"clauses": ["A", "B", "C"]},
            {"contradicts": False},
            {"contradicts": False},
            {"contradicts": False},
        ]
        from metrics.ircs import compute
        assert compute("q", "e") == 1.0

    @patch("metrics.ircs.llm_client.call_llm_json")
    def test_one_contradiction(self, mock_json):
        mock_json.side_effect = [
            {"clauses": ["A", "B", "C"]},  # 3 pairs
            {"contradicts": True},
            {"contradicts": False},
            {"contradicts": False},
        ]
        from metrics.ircs import compute
        score = compute("q", "e")
        assert abs(score - (1 - 1 / 3)) < 1e-6

    @patch("metrics.ircs.llm_client.call_llm_json")
    def test_single_clause(self, mock_json):
        mock_json.return_value = {"clauses": ["only one"]}
        from metrics.ircs import compute
        assert compute("q", "e") == 1.0


# ═══════════════════════════════════════════════════════════════
# 4. EDAS
# ═══════════════════════════════════════════════════════════════
class TestEDAS:
    @patch("metrics.edas.llm_client.call_llm_json")
    def test_strong_entailment(self, mock_json):
        mock_json.side_effect = [
            {"reasoning": "The user meets criteria", "decision": "Approved"},
            {"entailment_score": 0.95},
        ]
        from metrics.edas import compute
        assert compute("q", "e") == 0.95

    @patch("metrics.edas.llm_client.call_llm_json")
    def test_missing_decision(self, mock_json):
        mock_json.return_value = {"reasoning": "something", "decision": ""}
        from metrics.edas import compute
        assert compute("q", "e") == 0.0


# ═══════════════════════════════════════════════════════════════
# 5. SECS
# ═══════════════════════════════════════════════════════════════
class TestSECS:
    @patch("metrics.secs.llm_client.call_llm_json")
    def test_all_present(self, mock_json):
        mock_json.return_value = {
            "user_factors": True,
            "policy_rule": True,
            "logical_application": True,
            "decision_link": True,
        }
        from metrics.secs import compute
        assert compute("q", "e") == 1.0

    @patch("metrics.secs.llm_client.call_llm_json")
    def test_half_present(self, mock_json):
        mock_json.return_value = {
            "user_factors": True,
            "policy_rule": False,
            "logical_application": True,
            "decision_link": False,
        }
        from metrics.secs import compute
        assert compute("q", "e") == 0.5


# ═══════════════════════════════════════════════════════════════
# 6. PGSS
# ═══════════════════════════════════════════════════════════════
class TestPGSS:
    @patch("metrics.pgss.llm_client.call_llm_json")
    @patch("metrics.pgss._get_embedding_model")
    def test_all_supported(self, mock_model, mock_json):
        import numpy as np
        mock_json.return_value = {"clauses": ["clause1"]}
        # Make embeddings that produce high similarity
        encoder = MagicMock()
        encoder.encode.side_effect = [
            np.array([[1.0, 0.0]]),  # policy
            np.array([[1.0, 0.0]]),  # clause (identical)
        ]
        mock_model.return_value = encoder
        from metrics.pgss import compute
        assert compute("q", "e", policy_texts=["policy1"]) == 1.0

    def test_no_policies(self):
        from metrics.pgss import compute
        assert compute("q", "e", policy_texts=[]) == 0.0


# ═══════════════════════════════════════════════════════════════
# 7. ESI
# ═══════════════════════════════════════════════════════════════
class TestESI:
    @patch("metrics.esi._get_embedding_model")
    def test_identical_explanations(self, mock_model):
        import numpy as np
        encoder = MagicMock()
        encoder.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        mock_model.return_value = encoder
        from metrics.esi import compute
        score = compute("q", "e", precomputed_explanations=["a", "a", "a"])
        assert score == 1.0

    @patch("metrics.esi._get_embedding_model")
    def test_different_explanations(self, mock_model):
        import numpy as np
        encoder = MagicMock()
        encoder.encode.return_value = np.array([[1.0, 0.0], [-1.0, 0.0]])
        mock_model.return_value = encoder
        from metrics.esi import compute
        score = compute("q", "e", precomputed_explanations=["a", "b"])
        assert score < 0.1  # very different


# ═══════════════════════════════════════════════════════════════
# 8. EDR
# ═══════════════════════════════════════════════════════════════
class TestEDR:
    @patch("metrics.edr.llm_client.call_llm_json")
    def test_all_reasoning(self, mock_json):
        mock_json.return_value = {"reasoning_tokens": 50, "filler_tokens": 0}
        from metrics.edr import compute
        assert compute("q", "e") == 1.0

    @patch("metrics.edr.llm_client.call_llm_json")
    def test_half_filler(self, mock_json):
        mock_json.return_value = {"reasoning_tokens": 25, "filler_tokens": 25}
        from metrics.edr import compute
        assert compute("q", "e") == 0.5

    @patch("metrics.edr.llm_client.call_llm_json")
    def test_empty(self, mock_json):
        mock_json.return_value = {"reasoning_tokens": 0, "filler_tokens": 0}
        from metrics.edr import compute
        assert compute("q", "e") == 0.0
