"""
Metric 1 — Input Attribution Consistency Score (IACS)

Measures whether the explanation correctly reflects and uses
relevant attributes from the user query.

    IACS = |A_E ∩ A_Q| / |A_Q|
"""

import config
import llm_client
from llm_client import trace_collector

_EXTRACT_PROMPT = """\
You are an attribute extractor.  Given the TEXT below, list every
meaningful factual attribute (e.g. age, location, income, household
size, disability status, employment status, benefit type, etc.).

Return ONLY a JSON object: {{"attributes": ["attr1", "attr2", ...]}}

TEXT:
{text}"""


def _extract_attributes(text: str) -> list[str]:
    """Return a list of normalised attribute strings from *text*."""
    result = llm_client.call_llm_json(
        _EXTRACT_PROMPT.format(text=text),
        model=config.NLI_MODEL,
        caller="IACS",
    )
    attrs = result.get("attributes", [])
    return [a.strip().lower() for a in attrs if a.strip()]


def compute(query: str, explanation: str, **_kwargs) -> float:
    """
    Compute IACS for a query–explanation pair.

    Returns
    -------
    float
        Score in [0, 1].  1.0 means every query attribute is
        referenced in the explanation.
    """
    a_q = set(_extract_attributes(query))
    if not a_q:
        trace_collector.set_trace("IACS", {
            "formula": "IACS = |A_E ∩ A_Q| / |A_Q|",
            "note": "No attributes found in query → trivially consistent",
            "query_attributes": [],
            "explanation_attributes": [],
            "overlap": [],
            "score": 1.0,
        })
        return 1.0

    a_e = set(_extract_attributes(explanation))
    overlap = a_q & a_e
    score = len(overlap) / len(a_q)

    trace_collector.set_trace("IACS", {
        "formula": "IACS = |A_E ∩ A_Q| / |A_Q|",
        "query_attributes": sorted(a_q),
        "explanation_attributes": sorted(a_e),
        "overlap": sorted(overlap),
        "computation": f"|{sorted(overlap)}| / |{sorted(a_q)}| = {len(overlap)} / {len(a_q)} = {score:.4f}",
        "score": score,
    })

    return score
