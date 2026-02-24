"""
Persistent Vector Store.

JSON + NumPy based vector store with sentence-transformers embeddings.
Provides the same API as ChromaDB but works with Python 3.14.

Persistence: documents/metadata in JSON, embeddings in .npy file.
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

import config

STORE_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_FILE = os.path.join(STORE_DIR, "documents.json")
EMBEDS_FILE = os.path.join(STORE_DIR, "embeddings.npy")

_model = None
_documents: list[dict] | None = None
_embeddings: np.ndarray | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _ensure_dir():
    os.makedirs(STORE_DIR, exist_ok=True)


def _load():
    """Load documents and embeddings from disk into memory."""
    global _documents, _embeddings
    if _documents is not None:
        return

    if os.path.exists(DOCS_FILE) and os.path.exists(EMBEDS_FILE):
        with open(DOCS_FILE) as f:
            _documents = json.load(f)
        _embeddings = np.load(EMBEDS_FILE)
    else:
        _documents = []
        _embeddings = np.array([])


def _save():
    """Persist documents and embeddings to disk."""
    _ensure_dir()
    with open(DOCS_FILE, "w") as f:
        json.dump(_documents, f)
    if _embeddings is not None and len(_embeddings) > 0:
        np.save(EMBEDS_FILE, _embeddings)


def ingest(chunks: list[dict], batch_size: int = 100) -> int:
    """
    Ingest extracted PDF chunks into the vector store.

    Parameters
    ----------
    chunks : list[dict]
        Each dict must have ``text``, ``source``, ``category``,
        ``page``, ``chunk_index``.

    Returns
    -------
    int
        Total number of documents ingested.
    """
    global _documents, _embeddings

    model = _get_model()
    texts = [c["text"] for c in chunks]

    print(f"  🔢 Encoding {len(texts)} chunks…")
    all_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, batch_size=batch_size)

    _documents = chunks
    _embeddings = all_embeddings
    _save()

    print(f"\n✅ Vector store now has {len(_documents)} documents")
    return len(_documents)


def query(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Query the vector store for the most relevant policy chunks.

    Parameters
    ----------
    query_text : str
        The user's query.
    top_k : int
        Number of results to return.

    Returns
    -------
    list[dict]
        Each dict has ``text``, ``source``, ``category``, ``page``, ``distance``.
    """
    _load()
    if not _documents or _embeddings is None or len(_embeddings) == 0:
        return []

    model = _get_model()
    q_emb = model.encode([query_text], convert_to_numpy=True)[0]

    # Cosine similarity
    norms = np.linalg.norm(_embeddings, axis=1) * np.linalg.norm(q_emb)
    norms = np.where(norms == 0, 1e-10, norms)
    sims = np.dot(_embeddings, q_emb) / norms

    top_k = min(top_k, len(_documents))
    top_indices = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_indices:
        doc = _documents[idx]
        results.append({
            "text": doc["text"],
            "source": doc.get("source", ""),
            "category": doc.get("category", ""),
            "page": doc.get("page", 0),
            "distance": float(1 - sims[idx]),
        })
    return results


def get_all_documents() -> list[str]:
    """Return all document texts in the store (for PGSS scoring)."""
    _load()
    if not _documents:
        return []
    return [d["text"] for d in _documents]


def get_collection_stats() -> dict:
    """Return basic stats about the store."""
    _load()
    return {
        "collection_name": "policy_documents",
        "document_count": len(_documents) if _documents else 0,
        "persist_dir": STORE_DIR,
    }
