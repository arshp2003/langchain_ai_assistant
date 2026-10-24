"""Validation that compressed context still answers the query.

After compression, we need to verify the compressed text still contains
enough information to answer the query. If compression was too aggressive
and removed key facts, we fall back to the trimmed (pre-compression) context.

Approach: Compare embedding similarity between query and context before/after
compression. If compression drops similarity by more than a threshold, the
compressed version lost too much information — use the trimmed version instead.

Why embedding similarity, not LLM validation:
  - LLM validation would cost an extra API request (expensive on free tier:
    50 req/day, agent already burns ~6 per question)
  - Embedding similarity is cheap (local model, ~10ms) and already loaded
  - Semantic similarity correlates with "does this context answer the query?"
    well enough for a safety check

Threshold rationale:
  - Typical query-trimmed_context similarity: 0.6-0.8
  - Typical query-compressed_context similarity: 0.5-0.7 (slightly lower due
    to information loss, but still high)
  - If similarity drops by >0.15, compression likely removed key facts
  - Threshold is conservative: better to keep slightly longer context than
    lose answer-relevant information
"""

from embeddings import embeddings


# Maximum acceptable drop in cosine similarity after compression.
# If compressed context similarity is more than this much lower than trimmed
# context similarity, fall back to trimmed.
MAX_SIMILARITY_DROP = 0.15


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def validate_compression(
    query: str,
    trimmed_docs: list[tuple],
    compressed_docs: list[tuple],
) -> list[tuple]:
    """Check if compressed context still answers the query.

    Args:
        query: The original search query
        trimmed_docs: Context after trimming (before compression)
        compressed_docs: Context after compression

    Returns:
        The better of trimmed_docs or compressed_docs based on embedding
        similarity to the query. If validation fails (exception, empty input),
        returns trimmed_docs as the safe default.
    """
    if not trimmed_docs or not compressed_docs:
        return trimmed_docs

    try:
        # Concatenate all chunk texts for each version
        trimmed_text = " ".join(doc.page_content for doc, _ in trimmed_docs)
        compressed_text = " ".join(doc.page_content for doc, _ in compressed_docs)

        # Embed query and both context versions
        query_vec = embeddings.embed_query(query)
        trimmed_vec = embeddings.embed_query(trimmed_text)
        compressed_vec = embeddings.embed_query(compressed_text)

        # Compute similarities
        trimmed_sim = _cosine_similarity(query_vec, trimmed_vec)
        compressed_sim = _cosine_similarity(query_vec, compressed_vec)

        print(f"[validator] similarity: trimmed={trimmed_sim:.3f}, compressed={compressed_sim:.3f}")

        # If compression dropped similarity too much, fall back
        drop = trimmed_sim - compressed_sim
        if drop > MAX_SIMILARITY_DROP:
            print(f"[validator] compression dropped similarity by {drop:.3f} (>{MAX_SIMILARITY_DROP}), using trimmed context")
            return trimmed_docs

        print(f"[validator] compression preserved similarity (drop={drop:.3f}), using compressed context")
        return compressed_docs

    except Exception as exc:
        print(f"[validator] validation failed, using trimmed context: {exc}")
        return trimmed_docs
