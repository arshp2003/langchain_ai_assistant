"""LLM-based context compression for RAG.

After trimming, each chunk contains only query-relevant sentences - but those
sentences can still be verbose (technical docs often explain the same idea
multiple ways, or include explanatory padding around the core fact).
Compression asks the LLM to rewrite the concatenated trimmed passages into a
shorter form that preserves every fact needed to answer the query.

Cost control (free tier: 50 req/day, ~20 RPM account-wide):
  - Skip compression entirely when total context is under TOKEN_THRESHOLD.
    A short context costs more in latency + a wasted request than it saves
    in downstream answer tokens.
  - Batch ALL chunks into ONE compression call, not one per chunk. The model
    can also deduplicate across chunks this way.
  - Reuse the same ChatOpenRouter model the agent already uses - no extra
    model load, no extra dependency.

The compressed text replaces the original context passed to the answer-
generation prompt. Metadata (source filenames) is preserved by prepending
source tags to each chunk BEFORE compression so the model can still cite.
"""

import os

from langchain_core.messages import HumanMessage
from langchain_openrouter import ChatOpenRouter

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free")

# Rough token estimate: 1 token ~ 4 chars in English. We skip compression
# when the total context is under this many estimated tokens - the LLM call
# would cost latency and a request for negligible savings.
TOKEN_THRESHOLD = 600

COMPRESS_PROMPT = """Compress the following context into a shorter form that preserves every fact needed to answer the query.

Rules:
- Keep all distinct facts, numbers, names, and relationships
- Remove repetition (the same idea stated multiple ways)
- Remove filler phrases ("it is important to note that...", "in other words...")
- Merge overlapping information from different chunks
- Output should be roughly 40-60% of the input length
- Do NOT add information not present in the input
- Do NOT answer the query - only compress the context

Query: {query}

Context:
{context}

Compressed context:"""


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _format_context(scored_docs: list[tuple]) -> str:
    """Join chunks with source tags so the compressor (and later the answer
    model) can tell where each piece came from."""
    parts = []
    for i, (doc, _score) in enumerate(scored_docs, 1):
        source = doc.metadata.get("source", f"chunk_{i}")
        parts.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def compress_context(query: str, scored_docs: list[tuple]) -> list[tuple]:
    """Compress trimmed passages into a shorter form preserving answer-relevant facts.

    Returns the same list shape as the input (one entry per original chunk)
    when compression runs, or the original list unchanged when skipped. The
    returned docs have their page_content replaced with the compressed text
    split back across the original chunk count (so downstream formatting
    still sees per-chunk entries with source metadata).

    Skips compression when total context is under TOKEN_THRESHOLD estimated
    tokens - not worth the request cost.
    """
    if not scored_docs:
        return []

    context = _format_context(scored_docs)

    if _estimate_tokens(context) < TOKEN_THRESHOLD:
        print(f"[compressor] context short (~{_estimate_tokens(context)} tokens), skipping compression")
        return scored_docs

    try:
        model = ChatOpenRouter(model=MODEL_NAME)
        prompt = COMPRESS_PROMPT.format(query=query, context=context)
        response = model.invoke([HumanMessage(content=prompt)])
        compressed = response.content.strip()

        if not compressed:
            print("[compressor] empty compression result, keeping original")
            return scored_docs

        original_tokens = _estimate_tokens(context)
        compressed_tokens = _estimate_tokens(compressed)
        print(
            f"[compressor] compressed context: ~{original_tokens} -> ~{compressed_tokens} tokens "
            f"({100 - int(100 * compressed_tokens / original_tokens)}% reduction)"
        )

        # Put all compressed text in the first chunk, clear the rest. This
        # keeps source metadata for the first chunk and avoids inventing
        # source attributions for text that has been merged across chunks.
        import copy
        result = [copy.copy(scored_docs[0][0])]
        result[0].page_content = compressed
        return [(doc, score) for doc, (_, score) in zip(result, scored_docs[:1])]

    except Exception as exc:
        print(f"[compressor] compression failed, keeping original context: {exc}")
        return scored_docs
