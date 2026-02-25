"""
End-to-end Pipeline.

Query → RAG → Compute 8 metrics → Aggregate → Log → Return result.

Supports:
  - Single query: python pipeline.py "your query here"
  - Structured colored output with LLM analytics
"""

import logging
import time
from aggregator import compute_all_metrics
from monitor import log_interaction
import rag_pipeline
import pretty_print

logger = logging.getLogger(__name__)


def evaluate(query: str, *, silent: bool = False) -> dict:
    """
    Run the full explainability evaluation pipeline.

    1. Generate an explanation via RAG.
    2. Compute all 8 explainability metrics.
    3. Aggregate into a single score.
    4. Log the interaction.
    5. Return the structured result.

    Parameters
    ----------
    query : str
        The citizen's query.
    silent : bool
        If True, skip pretty-printing (useful for batch/API mode).

    Returns
    -------
    dict
        {
            "query": str,
            "explanation": str,
            "metric_scores": {…},
            "aggregate_score": float,
            "alerts": […],
            "metric_timings": {…},
            "metric_failures": {…},
            "llm_stats": {…},
            "llm_global": {…},
        }
    """
    pipeline_start = time.time()

    logger.info("Pipeline: generating explanation for query")
    context_chunks = rag_pipeline.retrieve(query)
    explanation = rag_pipeline.generate(query, context_chunks)

    logger.info("Pipeline: computing metrics")
    result = compute_all_metrics(
        query=query,
        explanation=explanation,
        policy_texts=rag_pipeline.get_policy_texts(),
        context="\n".join(context_chunks),
    )

    log_interaction(result, query=query, explanation=explanation)

    pipeline_time = round(time.time() - pipeline_start, 3)

    full_result = {
        "query": query,
        "explanation": explanation,
        "pipeline_time": pipeline_time,
        **result,
    }

    if not silent:
        pretty_print.print_result(full_result, query=query)

    return full_result


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.WARNING)
    q = " ".join(sys.argv[1:]) or "Am I eligible for council housing? I am 25 and have two children."
    evaluate(q)
