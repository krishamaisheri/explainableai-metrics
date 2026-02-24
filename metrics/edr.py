"""
Metric 8 — Explanation Density Ratio (EDR)

Measures the proportion of explanation tokens devoted to
substantive reasoning versus filler / boilerplate.

    EDR = reasoning_tokens / total_tokens
"""

import llm_client

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
        _CLASSIFY_PROMPT.format(explanation=explanation)
    )
    reasoning = int(result.get("reasoning_tokens", 0))
    filler = int(result.get("filler_tokens", 0))
    total = reasoning + filler
    if total == 0:
        return 0.0
    return reasoning / total
