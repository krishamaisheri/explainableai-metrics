"""
Metric 6 — Policy Grounding Similarity Score (PGSS)

Measures semantic support of each explanation clause by the
policy corpus using embedding cosine similarity.

    PGSS = (# supported clauses) / (total clauses)
"""

import numpy as np
from sentence_transformers import SentenceTransformer

import config
import llm_client

_model = None


def _get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


_SEGMENT_PROMPT = """\
Segment the following EXPLANATION into individual factual or
reasoning clauses.  Each clause should be a single statement.

Return ONLY a JSON object: {{"clauses": ["clause1", "clause2", ...]}}

EXPLANATION:
{explanation}"""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def compute(
    _query: str,
    explanation: str,
    *,
    policy_texts: list[str] | None = None,
    **_kwargs,
) -> float:
    """
    Compute PGSS for an explanation against the policy corpus.

    Parameters
    ----------
    policy_texts : list[str]
        List of policy document passages / chunks.

    Returns
    -------
    float
        Score in [0, 1].  Fraction of clauses semantically
        supported (≥ threshold) by the policy corpus.
    """
    if not policy_texts:
        return 0.0

    # Segment explanation into clauses
    seg = llm_client.call_llm_json(
        _SEGMENT_PROMPT.format(explanation=explanation)
    )
    clauses = seg.get("clauses", [])
    if not clauses:
        return 1.0

    model = _get_embedding_model()
    policy_embeddings = model.encode(policy_texts, convert_to_numpy=True)
    clause_embeddings = model.encode(clauses, convert_to_numpy=True)

    threshold = config.PGSS_SIMILARITY_THRESHOLD
    supported = 0

    for c_emb in clause_embeddings:
        max_sim = max(
            _cosine_similarity(c_emb, p_emb) for p_emb in policy_embeddings
        )
        if max_sim >= threshold:
            supported += 1

    return supported / len(clauses)
