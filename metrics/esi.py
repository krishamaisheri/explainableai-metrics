"""
Metric 7 — Explanation Stability Index (ESI)

Measures variance of explanation across repeated generation runs.

    ESI = 1 − average_cosine_distance(repeated_outputs)
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


_GENERATION_PROMPT = """\
Answer the following user query based on the provided context.
Give a clear, structured explanation for your answer.

QUERY:
{query}

CONTEXT:
{context}"""


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 1.0
    return 1.0 - float(dot / norm)


def compute(
    query: str,
    explanation: str,
    *,
    context: str = "",
    repeat_runs: int | None = None,
    precomputed_explanations: list[str] | None = None,
    **_kwargs,
) -> float:
    """
    Compute ESI for an explanation.

    If *precomputed_explanations* is provided (e.g. from tests),
    those are used directly.  Otherwise, the generator is called
    *repeat_runs* times to produce variant explanations.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means perfectly stable explanations.
    """
    runs = repeat_runs or config.ESI_REPEAT_RUNS

    if precomputed_explanations is not None:
        explanations = precomputed_explanations
    else:
        explanations = [explanation]  # include the original
        for _ in range(runs - 1):
            resp = llm_client.call_llm(
                _GENERATION_PROMPT.format(query=query, context=context)
            )
            explanations.append(resp)

    if len(explanations) < 2:
        return 1.0

    model = _get_embedding_model()
    embeddings = model.encode(explanations, convert_to_numpy=True)

    distances = []
    n = len(embeddings)
    for i in range(n):
        for j in range(i + 1, n):
            distances.append(_cosine_distance(embeddings[i], embeddings[j]))

    avg_distance = float(np.mean(distances)) if distances else 0.0
    return max(0.0, 1.0 - avg_distance)
