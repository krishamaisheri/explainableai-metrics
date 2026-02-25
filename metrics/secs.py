"""
Metric 5 — Structured Explanation Completeness Score (SECS)

Detects presence of four required explanation components:
  1. User factors
  2. Policy rule
  3. Logical application
  4. Decision link

    SECS = (# components present) / 4
"""

import config
import llm_client
from llm_client import trace_collector

_COMPONENTS = ["user_factors", "policy_rule", "logical_application", "decision_link"]

_DETECT_PROMPT = """\
Analyse the EXPLANATION below and determine which of the following
structural components are present:

1. user_factors — References to the user's specific circumstances
2. policy_rule — Citation or paraphrase of the relevant policy/rule
3. logical_application — Logical application of the rule to the user's case
4. decision_link — Clear link between the reasoning and the final decision

Return ONLY a JSON object mapping each component to true/false:
{{
  "user_factors": true,
  "policy_rule": false,
  "logical_application": true,
  "decision_link": true
}}

EXPLANATION:
{explanation}"""


def compute(_query: str, explanation: str, **_kwargs) -> float:
    """
    Compute SECS for an explanation.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means all four structural
        components are present.
    """
    result = llm_client.call_llm_json(
        _DETECT_PROMPT.format(explanation=explanation),
        model=config.NLI_MODEL,
        caller="SECS",
    )

    component_results = {}
    present = 0
    for c in _COMPONENTS:
        val = result.get(c)
        if isinstance(val, str):
            is_present = val.lower() == "true"
        else:
            is_present = bool(val)
        component_results[c] = is_present
        if is_present:
            present += 1

    score = present / len(_COMPONENTS)

    trace_collector.set_trace("SECS", {
        "formula": "SECS = (# components present) / 4",
        "components_checked": _COMPONENTS,
        "component_results": component_results,
        "present_count": present,
        "total_components": len(_COMPONENTS),
        "computation": f"{present} / {len(_COMPONENTS)} = {score:.4f}",
        "score": score,
    })

    return score
