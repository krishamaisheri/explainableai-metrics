"""
Metric 2 — Input Contradiction Rate (ICR)

Detects whether the explanation contradicts factual statements
made in the user query.

    ICR_Score = 1 − (contradictions / total_facts)
"""

import llm_client

_EXTRACT_FACTS_PROMPT = """\
Extract every factual statement from the QUERY below.
Return ONLY a JSON object: {{"facts": ["fact1", "fact2", ...]}}

QUERY:
{query}"""

_CONTRADICTION_PROMPT = """\
You are a contradiction detector.
Given a FACT from the user's query and an EXPLANATION from the system,
determine whether the explanation CONTRADICTS the fact.

Reply ONLY with a JSON object: {{"contradicts": true}} or {{"contradicts": false}}

FACT:
{fact}

EXPLANATION:
{explanation}"""


def compute(query: str, explanation: str, **_kwargs) -> float:
    """
    Compute ICR score for a query–explanation pair.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means no contradictions detected.
    """
    facts_result = llm_client.call_llm_json(
        _EXTRACT_FACTS_PROMPT.format(query=query)
    )
    facts = facts_result.get("facts", [])
    if not facts:
        return 1.0

    contradictions = 0
    for fact in facts:
        result = llm_client.call_llm_json(
            _CONTRADICTION_PROMPT.format(fact=fact, explanation=explanation),
            model=None,  # uses default NLI_MODEL via config
        )
        if result.get("contradicts", False):
            contradictions += 1

    return 1.0 - (contradictions / len(facts))
