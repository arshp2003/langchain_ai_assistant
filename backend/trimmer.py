"""Passage trimming for RAG search results.

Chunk-level filtering (see reranker.filter_relevant) drops chunks that are
irrelevant overall, but a *kept* chunk can still be mostly filler - e.g. a
500-word chunk where only two sentences actually answer the query. Trimming
scores every sentence against the query with the same cross-encoder and
keeps only the relevant ones.

Rules per chunk:
  - keep sentences scoring within `margin` of the chunk's best sentence.
    A relative margin is used because the cross-encoder's absolute score
    scale is not comparable across text lengths (a chunk scores ~+7 where
    its best sentence scores ~+0.7), but its RANKING stays reliable.
  - floor: the best sentence is always kept (it is trivially within margin
    of itself), so a chunk that passed chunk-level filtering never empties
  - cap: at most max_sentences kept, chosen by score rank so the cap can
    never evict the best sentence
  - kept sentences are restored to their original document order so the
    text still reads naturally

All sentences across all chunks are scored in ONE batched cross-encoder
call (~15-20 pairs, tens of milliseconds) rather than one call per chunk.
"""

import copy
import re

from reranker import _get_cross_encoder

# Split on sentence-ending punctuation followed by whitespace and a new
# sentence start (capital letter, digit, or opening quote/paren).
_SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])')

DEFAULT_MARGIN = 1.0
DEFAULT_MAX_SENTENCES = 4


def split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def trim_passages(
    query: str,
    scored_docs: list[tuple],
    margin: float = DEFAULT_MARGIN,
    max_sentences: int = DEFAULT_MAX_SENTENCES,
) -> list[tuple]:
    """Trim each (doc, score) chunk down to its query-relevant sentences.

    Args:
        query: The original search query sentences are scored against
        scored_docs: Output of rerank()/filter_relevant() - (doc, chunk_score) tuples
        margin: Keep sentences scoring within this many points of the
            chunk's best sentence
        max_sentences: Hard cap on sentences kept per chunk

    Returns:
        List of (doc, chunk_score) tuples where doc.page_content is the
        trimmed text. Chunks with 1 sentence (or trimming failure) pass
        through unchanged.
    """
    if not scored_docs:
        return []

    try:
        per_chunk_sentences = []
        batch_pairs = []

        for doc, chunk_score in scored_docs:
            sentences = split_sentences(doc.page_content)
            per_chunk_sentences.append(sentences)
            batch_pairs.extend((query, s) for s in sentences)

        model = _get_cross_encoder()
        scores = model.predict(batch_pairs) if batch_pairs else []
    except Exception as exc:
        print(f"[trimmer] trimming failed, keeping chunks as-is: {exc}")
        return list(scored_docs)

    trimmed = []
    cursor = 0
    for (doc, chunk_score), sentences in zip(scored_docs, per_chunk_sentences):
        n = len(sentences)
        sent_scores = list(map(float, scores[cursor:cursor + n]))
        cursor += n

        if n <= 1:
            trimmed.append((doc, chunk_score))
            continue

        by_score = sorted(range(n), key=lambda i: sent_scores[i], reverse=True)
        best_score = sent_scores[by_score[0]]

        # Relative margin: anything close to the best sentence is relevant
        # too. The best sentence itself always qualifies (floor).
        keep = {i for i in by_score if sent_scores[i] >= best_score - margin}
        # Cap by score rank so the cap can never evict the best sentence.
        keep = set(by_score[:max_sentences]) & keep

        keep_idx = sorted(keep)  # restore original document order

        kept_text = " ".join(sentences[i] for i in keep_idx)
        if kept_text == doc.page_content.strip():
            trimmed.append((doc, chunk_score))
            continue

        trimmed_doc = copy.copy(doc)
        trimmed_doc.page_content = kept_text
        dropped = n - len(keep_idx)
        print(f"[trimmer] chunk trimmed: kept {len(keep_idx)}/{n} sentences (dropped {dropped})")
        trimmed.append((trimmed_doc, chunk_score))

    return trimmed
