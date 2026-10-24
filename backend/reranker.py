"""Cross-encoder reranker for RAG search results.

Vector search ranks chunks by embedding similarity - fast but rough, since
each chunk is compared as a single vector against the query vector. A
cross-encoder instead reads every (query, chunk) pair together through a
transformer and scores true relevance, which is slower but far more accurate.

The pipeline: vector search fetches a wide pool of candidates (k=10),
then this reranker re-scores them and keeps only the best few.

The model is lazy-loaded on first use (thread-safe) so server startup
stays fast - same pattern as the local embeddings model.
"""

import threading

from sentence_transformers import CrossEncoder

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder: CrossEncoder | None = None
_lock = threading.Lock()


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        with _lock:
            if _cross_encoder is None:
                _cross_encoder = CrossEncoder(MODEL_NAME)
    return _cross_encoder


def rerank(query: str, docs: list, top_k: int = 3) -> list[tuple]:
    """Re-score documents against the query with a cross-encoder.

    Args:
        query: The search query
        docs: List of LangChain Document objects retrieved from vector search
        top_k: How many top-ranked documents to return

    Returns:
        List of (document, score) tuples sorted by relevance score descending,
        truncated to top_k. On failure, returns the original order so search
        degrades gracefully instead of breaking.
    """
    if not docs:
        return []

    try:
        model = _get_cross_encoder()
        pairs = [(query, doc.page_content) for doc in docs]
        scores = model.predict(pairs)
        scored = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        return scored[:top_k]
    except Exception as exc:
        print(f"[reranker] reranking failed, using vector-search order: {exc}")
        return [(doc, 0.0) for doc in docs[:top_k]]


# Cross-encoder score scale (ms-marco-MiniLM-L-6-v2 logits):
#   > 5 highly relevant, 1-5 relevant, 0-1 borderline, < 0 irrelevant.
# 1.0 cleanly separates "answers the question" from "merely mentions it".
DEFAULT_MIN_SCORE = 1.0


def filter_relevant(scored_docs: list[tuple], min_score: float = DEFAULT_MIN_SCORE) -> list[tuple]:
    """Drop documents whose reranker score falls below min_score.

    Returns an empty list when nothing is relevant. Callers should treat
    that as "no useful context" rather than feeding junk to the LLM, which
    invites hallucinated answers.
    """
    filtered = [(doc, score) for doc, score in scored_docs if score >= min_score]
    dropped = len(scored_docs) - len(filtered)
    if dropped:
        print(f"[reranker] relevance filter dropped {dropped} chunk(s) below {min_score}")
    return filtered
