"""
Metric 2 — Input Contradiction Rate (ICR)

Detects whether the explanation contradicts factual statements
made in the user query.

    ICR_Score = 1 − (contradictions / total_facts)
"""

import config
import llm_client
from llm_client import trace_collector

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
        _EXTRACT_FACTS_PROMPT.format(query=query),
        model=config.NLI_MODEL,
        caller="ICR",
    )
    facts = facts_result.get("facts", [])
    if not facts:
        trace_collector.set_trace("ICR", {
            "formula": "ICR = 1 − (contradictions / total_facts)",
            "note": "No facts extracted from query",
            "facts": [],
            "fact_checks": [],
            "contradictions": 0,
            "score": 1.0,
        })
        return 1.0

    contradictions = 0
    fact_checks = []

    for fact in facts:
        result = llm_client.call_llm_json(
            _CONTRADICTION_PROMPT.format(fact=fact, explanation=explanation),
            model=config.NLI_MODEL,
            caller="ICR",
        )
        val = result.get("contradicts")

        if isinstance(val, str):
            is_contradiction = val.lower() == "true"
        else:
            is_contradiction = bool(val)

        fact_checks.append({
            "fact": fact,
            "contradicts": is_contradiction,
        })

        if is_contradiction:
            contradictions += 1

    score = 1.0 - (contradictions / len(facts))

    trace_collector.set_trace("ICR", {
        "formula": "ICR = 1 − (contradictions / total_facts)",
        "facts": facts,
        "fact_checks": fact_checks,
        "contradictions": contradictions,
        "total_facts": len(facts),
        "computation": f"1 − ({contradictions} / {len(facts)}) = {score:.4f}",
        "score": score,
    })

    return score
