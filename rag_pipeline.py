"""
RAG Pipeline — ChromaDB Edition.

Query → Retrieve top-k policy chunks from ChromaDB → Generate explanation.
"""

import config
import vector_store
import llm_client


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Retrieve the top-k policy chunks most relevant to *query* from ChromaDB."""
    results = vector_store.query(query, top_k=top_k)
    return [r["text"] for r in results]


def generate(query: str, context_chunks: list[str] | None = None) -> str:
    """
    Generate a policy-grounded explanation for a user query.

    Parameters
    ----------
    query : str
        The citizen's query.
    context_chunks : list[str], optional
        Pre-retrieved policy chunks.  If None, retrieval is run.
    """
    if context_chunks is None:
        context_chunks = retrieve(query)

    context = "\n".join(f"- {c}" for c in context_chunks)

    prompt = f"""\
You are a UK public-sector assistant.  Answer the citizen's query
using ONLY the policy information provided.  Structure your answer
with:
1. Relevant user factors
2. Applicable policy rule(s)
3. How the rule applies to this user's case
4. Your decision / recommendation

POLICY CONTEXT:
{context}

CITIZEN QUERY:
{query}

ANSWER:"""

    return llm_client.call_llm(
        prompt,
        model=config.REASONING_MODEL,
        caller="RAG_GENERATION",
    )


def get_policy_texts() -> list[str]:
    """Return all policy chunks from ChromaDB (for PGSS scoring)."""
    return vector_store.get_all_documents()
