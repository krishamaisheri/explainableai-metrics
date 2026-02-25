"""
Metric 3 — Internal Reasoning Consistency Score (IRCS)

Checks whether reasoning statements within the explanation
contradict each other.

    IRCS = 1 − T / P
    where T = contradictory pairs, P = k(k−1)/2
"""

import llm_client

_SEGMENT_PROMPT = """\
Segment the following EXPLANATION into individual reasoning clauses.
Each clause should be a single, self-contained logical statement.

Return ONLY a JSON object: {{"clauses": ["clause1", "clause2", ...]}}

EXPLANATION:
{explanation}"""

_PAIR_CONTRADICTION_PROMPT = """\
You are a contradiction detector.
Determine whether the following two statements contradict each other.

Reply ONLY with a JSON object: {{"contradicts": true}} or {{"contradicts": false}}

STATEMENT A:
{clause_a}

STATEMENT B:
{clause_b}"""


def compute(_query: str, explanation: str, **_kwargs) -> float:
    """
    Compute IRCS for an explanation.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means no internal contradictions.
    """
    seg_result = llm_client.call_llm_json(
        _SEGMENT_PROMPT.format(explanation=explanation)
    )
    clauses = seg_result.get("clauses", [])
    k = len(clauses)
    if k < 2:
        return 1.0  # can't contradict with fewer than 2 clauses

    total_pairs = k * (k - 1) // 2
    contradictory = 0

    for i in range(k):
        for j in range(i + 1, k):
            result = llm_client.call_llm_json(
                _PAIR_CONTRADICTION_PROMPT.format(
                    clause_a=clauses[i], clause_b=clauses[j]
                )
            )
            val = result.get("contradicts")
            
            # Robust boolean conversion
            if isinstance(val, str):
                is_contradiction = val.lower() == "true"
            else:
                is_contradiction = bool(val)

            if is_contradiction:
                contradictory += 1

    return 1.0 - (contradictory / total_pairs)
