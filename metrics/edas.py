"""
Metric 4 — Explanation–Decision Alignment Score (EDAS)

Measures whether the reasoning in the explanation logically
entails the final decision.

    EDAS = P(Decision | Reasoning)   (entailment probability)
"""

import config
import llm_client
from llm_client import trace_collector

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
        _EXTRACT_PROMPT.format(explanation=explanation),
        model=config.NLI_MODEL,
        caller="EDAS",
    )
    reasoning = parts.get("reasoning", "")
    decision = parts.get("decision", "")

    if not reasoning or not decision:
        trace_collector.set_trace("EDAS", {
            "formula": "EDAS = P(Decision | Reasoning)  (entailment probability)",
            "note": "Could not extract reasoning or decision from explanation",
            "extracted_reasoning": reasoning,
            "extracted_decision": decision,
            "entailment_score": 0.0,
            "score": 0.0,
        })
        return 0.0

    result = llm_client.call_llm_json(
        _ENTAILMENT_PROMPT.format(reasoning=reasoning, decision=decision),
        model=config.NLI_MODEL,
        caller="EDAS",
    )
    raw_score = float(result.get("entailment_score", 0.0))
    score = max(0.0, min(1.0, raw_score))

    trace_collector.set_trace("EDAS", {
        "formula": "EDAS = P(Decision | Reasoning)  (entailment probability)",
        "extracted_reasoning": reasoning[:300],
        "extracted_decision": decision[:300],
        "entailment_score": raw_score,
        "clamped_score": score,
        "computation": f"entailment_score = {raw_score} → clamped to [0,1] = {score:.4f}",
        "score": score,
    })

    return score
