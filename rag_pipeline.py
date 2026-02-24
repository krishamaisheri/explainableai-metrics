"""
Minimal RAG Pipeline.

Query → Retrieve top-k policy chunks → Generate explanation.
Uses sentence-transformers for retrieval and OpenRouter for generation.
"""

import os
import glob
import numpy as np
from sentence_transformers import SentenceTransformer

import config
import llm_client

_embed_model = None
_policy_chunks: list[str] = []
_policy_embeddings: np.ndarray | None = None

POLICY_DIR = os.path.join(os.path.dirname(__file__), "sample_policies")


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embed_model


def _load_policies() -> list[str]:
    """Load and chunk sample policy documents."""
    global _policy_chunks, _policy_embeddings
    if _policy_chunks:
        return _policy_chunks

    chunks = []
    for path in sorted(glob.glob(os.path.join(POLICY_DIR, "*.txt"))):
        with open(path) as f:
            text = f.read().strip()
        # Chunk by paragraph / numbered item
        for para in text.split("\n"):
            para = para.strip()
            if para:
                chunks.append(para)

    _policy_chunks = chunks
    model = _get_embed_model()
    _policy_embeddings = model.encode(chunks, convert_to_numpy=True)
    return chunks


def retrieve(query: str, top_k: int = 5) -> list[str]:
    """Retrieve the top-k policy chunks most relevant to *query*."""
    _load_policies()
    model = _get_embed_model()
    q_emb = model.encode([query], convert_to_numpy=True)[0]

    sims = np.dot(_policy_embeddings, q_emb) / (
        np.linalg.norm(_policy_embeddings, axis=1) * np.linalg.norm(q_emb) + 1e-10
    )
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [_policy_chunks[i] for i in top_indices]


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

    return llm_client.call_llm(prompt)


def get_policy_texts() -> list[str]:
    """Return all loaded policy chunks (for PGSS scoring)."""
    _load_policies()
    return list(_policy_chunks)
