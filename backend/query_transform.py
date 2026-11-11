"""Multi-query expansion: search several phrasings instead of betting on one.

Hybrid retrieval merges two methods, but both are fed the SAME query string. If
that one phrasing happens to miss the document's vocabulary, vector search and
BM25 fail together - hybrid buys term coverage, not phrasing coverage.

Expansion asks the model for a few differently-worded versions of the same
question, runs the existing hybrid search on each, and merges the candidate
pools. Three independent phrasings rarely miss the same way, so a chunk none of
them would have found alone can still surface.

The candidates are interleaved rather than concatenated before the cap, so every
variant contributes its best matches before any variant's weaker tail is dropped.
Ranking is left entirely to the cross-encoder downstream - the variants are not
scored on a comparable scale and must not be fused by score.

Costs one extra model request, which matters on a free-tier cap, so callers gate
this on a weak first pass rather than running it on every search (see
`search_documents` in agent_graph.py). Failures fall back to the original query
alone so expansion can never block retrieval.
"""

import re

from langchain_core.messages import HumanMessage

N_VARIANTS = 3

# Cap on the merged candidate pool handed to the reranker. Sized so the
# cross-encoder stays in the tens-of-milliseconds range even with several
# variants contributing.
MAX_CANDIDATES = 30

_EXPANSION_PROMPT = """Generate {n} alternative phrasings of the search query below. \
Each must target the same information but use different words, so a document retriever \
would match a different vocabulary each time.

Query: {query}

Rules:
- Keep the topic and intent identical; only change the wording.
- Prefer synonyms, related technical terms, and alternate spellings of acronyms.
- Do NOT answer the question and do NOT invent facts.
- Output exactly one phrasing per line, with no numbering, bullets, or commentary."""


def _parse_variants(text: str) -> list[str]:
    """Pull one phrasing per line out of the model's reply.

    Strips list markers the model adds despite instructions ("1.", "-", "*"),
    and drops anything too short to be a usable search string.
    """
    variants = []

    for line in text.splitlines():
        cleaned = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()

        if len(cleaned) < 4:
            continue
        variants.append(cleaned)

    return variants


def expand_queries(query: str, n_variants: int = N_VARIANTS) -> list[str]:
    """Return the original query plus model-generated rephrasings.

    The original always leads the list, so a failed or empty expansion still
    leaves a working search. Duplicate phrasings are dropped.
    """
    from agent_graph import invoke_model, model, text_of

    try:
        response = invoke_model(
            model,
            [HumanMessage(content=_EXPANSION_PROMPT.format(
                n=n_variants, query=query
            ))],
        )
    except Exception as exc:
        print(f"[query] expansion failed, using original query only: {exc}")
        return [query]

    variants = _parse_variants(text_of(response))

    queries = [query]
    for variant in variants:
        if variant.lower() == query.lower():
            continue
        if variant in queries:
            continue
        queries.append(variant)
        if len(queries) >= n_variants + 1:
            break

    print(f"[query] {len(queries)} phrasings: {queries}")
    return queries
