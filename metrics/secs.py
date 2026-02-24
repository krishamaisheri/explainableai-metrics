"""
Metric 5 — Structured Explanation Completeness Score (SECS)

Detects presence of four required explanation components:
  1. User factors
  2. Policy rule
  3. Logical application
  4. Decision link

    SECS = (# components present) / 4
"""

import llm_client

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
        _DETECT_PROMPT.format(explanation=explanation)
    )
    present = sum(1 for c in _COMPONENTS if result.get(c, False))
    return present / len(_COMPONENTS)
