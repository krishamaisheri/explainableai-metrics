"""
End-to-end Pipeline.

Query → RAG → Compute 8 metrics → Aggregate → Log → Return result.
"""

import logging
from aggregator import compute_all_metrics
from monitor import log_interaction
import rag_pipeline

logger = logging.getLogger(__name__)


def evaluate(query: str) -> dict:
    """
    Run the full explainability evaluation pipeline.

    1. Generate an explanation via RAG.
    2. Compute all 8 explainability metrics.
    3. Aggregate into a single score.
    4. Log the interaction.
    5. Return the structured result.

    Returns
    -------
    dict
        {
            "query": str,
            "explanation": str,
            "metric_scores": {…},
            "aggregate_score": float,
            "alerts": […],
        }
    """
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

    return {
        "query": query,
        "explanation": explanation,
        **result,
    }


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    q = " ".join(sys.argv[1:]) or "Am I eligible for council housing? I am 25 and have two children."
    result = evaluate(q)
    import json
    print(json.dumps(result, indent=2))
