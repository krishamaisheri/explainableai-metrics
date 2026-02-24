"""
Metric 4 — Explanation–Decision Alignment Score (EDAS)

Measures whether the reasoning in the explanation logically
entails the final decision.

    EDAS = P(Decision | Reasoning)   (entailment probability)
"""

import llm_client

_EXTRACT_PROMPT = """\
From the EXPLANATION below, extract:
1. The REASONING (the justification / logical argument).
2. The DECISION (the final conclusion or action recommended).

Return ONLY a JSON object:
{{"reasoning": "...", "decision": "..."}}

EXPLANATION:
{explanation}"""

_ENTAILMENT_PROMPT = """\
You are an entailment checker.
Given a REASONING passage and a DECISION, rate how strongly the
reasoning entails (logically leads to) the decision.

Return ONLY a JSON object with a score between 0.0 and 1.0:
{{"entailment_score": 0.85}}

REASONING:
{reasoning}

DECISION:
{decision}"""


def compute(_query: str, explanation: str, **_kwargs) -> float:
    """
    Compute EDAS for an explanation.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means perfect alignment between
        reasoning and decision.
    """
    parts = llm_client.call_llm_json(
        _EXTRACT_PROMPT.format(explanation=explanation)
    )
    reasoning = parts.get("reasoning", "")
    decision = parts.get("decision", "")

    if not reasoning or not decision:
        return 0.0

    result = llm_client.call_llm_json(
        _ENTAILMENT_PROMPT.format(reasoning=reasoning, decision=decision)
    )
    score = float(result.get("entailment_score", 0.0))
    return max(0.0, min(1.0, score))
