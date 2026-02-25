"""
Metric 8 — Explanation Density Ratio (EDR)

Measures the proportion of explanation tokens devoted to
substantive reasoning versus filler / boilerplate.

    EDR = reasoning_tokens / total_tokens
"""

import config
import llm_client
from llm_client import trace_collector

_CLASSIFY_PROMPT = """\
Analyse the EXPLANATION below.  Classify how many tokens (words)
are substantive reasoning (logical arguments, policy references,
factual analysis) versus filler (greetings, hedging language,
boilerplate disclaimers, pleasantries, repetition).

Return ONLY a JSON object:
{{"reasoning_tokens": 45, "filler_tokens": 12}}

EXPLANATION:
{explanation}"""


def compute(_query: str, explanation: str, **_kwargs) -> float:
    """
    Compute EDR for an explanation.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means entirely substantive reasoning.
    """
    result = llm_client.call_llm_json(
        _CLASSIFY_PROMPT.format(explanation=explanation),
        model=config.NLI_MODEL,
        caller="EDR",
    )
    reasoning = int(result.get("reasoning_tokens", 0))
    filler = int(result.get("filler_tokens", 0))
    total = reasoning + filler

    if total == 0:
        trace_collector.set_trace("EDR", {
            "formula": "EDR = reasoning_tokens / (reasoning_tokens + filler_tokens)",
            "note": "Total tokens = 0",
            "reasoning_tokens": 0,
            "filler_tokens": 0,
            "score": 0.0,
        })
        return 0.0

    score = reasoning / total

    trace_collector.set_trace("EDR", {
        "formula": "EDR = reasoning_tokens / (reasoning_tokens + filler_tokens)",
        "reasoning_tokens": reasoning,
        "filler_tokens": filler,
        "total_tokens": total,
        "computation": f"{reasoning} / ({reasoning} + {filler}) = {reasoning} / {total} = {score:.4f}",
        "score": score,
    })

    return score
