"""Hybrid retrieval: combine vector search with BM25 keyword search.

Vector search ranks by embedding similarity, which understands that "concurrent
writes" and "parallel transactions" are the same idea. But it can still miss a
chunk that repeats an exact technical term the user typed, because term frequency
barely moves the embedding.

BM25 is the opposite: pure term-frequency scoring with no semantics at all. It
finds every chunk that literally contains the user's words, which is exactly what
you want for acronyms (WAL, MVCC, RAG) and code identifiers.

Running both and merging their candidate pools recovers the recall each method
loses alone. We do NOT fuse the two score scales - they are not comparable
(embedding cosine sits in [0,1], BM25 is unbounded). Instead we take the union
and let the cross-encoder reranker score every candidate, since it is a far
better ranker than any fusion formula.

The BM25 index is built from the chunks already persisted in ChromaDB, so there
is no second copy of the corpus to maintain. It is cached and invalidated whenever
a document is uploaded or deleted.

Searches can be scoped to a set of filenames, and the scope is applied INSIDE
both retrievers rather than to their merged output: a nine-chunk resume cannot
outrank 232 unrelated chunks to reach the pool in the first place.
"""

import re
import threading

from langchain_core.documents import Document

# How many candidates each retriever contributes to the merged pool. Kept equal
# so neither method dominates the union before reranking.
VECTOR_K = 10
KEYWORD_K = 10

# Ceiling on the pool handed to the cross-encoder once every phrasing has been
# searched. Without it, three phrasings could push ~60 pairs through the
# reranker for a question that only needs its top 3.
MAX_CANDIDATES = 30

_bm25_retriever = None
_bm25_lock = threading.Lock()
_bm25_doc_count = 0
# Scoped indexes, keyed by the sorted tuple of filenames they cover. Separate
# from the whole-corpus index because a per-file index answers a different
# question and must not overwrite the general one.
_bm25_scoped: dict[tuple[str, ...], tuple[object, int]] = {}

# BM25 has no semantics, so it lives entirely on tokens the two sides share.
# The library default is str.split(), which keeps punctuation and case, so
# "MVCC." in a chunk and "MVCC" in a query were different words and never
# matched - measured, that made keyword search contribute almost nothing.
_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a word character."""
    return _TOKEN.findall(text.lower())


def _source_filter(sources) -> dict | None:
    """Chroma `where` clause for the requested files, or None for the whole corpus."""
    if not sources:
        return None
    return {"source": {"$in": sorted(sources)}}


def _load_all_documents(vector_store, sources=None) -> list[Document]:
    """Pull chunks out of the Chroma collection as LangChain Documents.

    Passing sources narrows the read at the database, so a search scoped to one
    file builds its keyword index over that file's chunks and nothing else.
    """
    raw = vector_store.get(where=_source_filter(sources))

    documents = []
    for doc_id, text, metadata in zip(
        raw["ids"], raw["documents"], raw["metadatas"]
    ):
        documents.append(
            Document(page_content=text, metadata=metadata or {}, id=doc_id)
        )
    return documents


def _build_bm25(vector_store, sources=None):
    """Build a BM25Retriever over the persisted chunks."""
    from langchain_community.retrievers import BM25Retriever

    documents = _load_all_documents(vector_store, sources)
    if not documents:
        return None

    retriever = BM25Retriever.from_documents(documents, preprocess_func=tokenize)
    retriever.k = KEYWORD_K
    return retriever


def get_bm25_retriever(vector_store, sources=None):
    """Return the cached BM25 retriever, rebuilding it if the corpus changed.

    With sources, the index covers only those files; without, the whole corpus.
    """
    global _bm25_retriever, _bm25_doc_count

    key = tuple(sorted(sources)) if sources else ()

    with _bm25_lock:
        current_count = vector_store._collection.count()

        if key:
            cached = _bm25_scoped.get(key)
            if cached is None or cached[1] != current_count:
                retriever = _build_bm25(vector_store, list(key))
                _bm25_scoped[key] = (retriever, current_count)
                print(
                    f"[hybrid] BM25 index built over {key} "
                    f"({current_count} chunks in collection)"
                )
            return _bm25_scoped[key][0]

        if _bm25_retriever is None or _bm25_doc_count != current_count:
            _bm25_retriever = _build_bm25(vector_store)
            _bm25_doc_count = current_count
            print(f"[hybrid] BM25 index built over {current_count} chunks")

        return _bm25_retriever


def invalidate_bm25() -> None:
    """Drop the cached indexes so the next search rebuilds them.

    Call after uploading or deleting a document, otherwise keyword search keeps
    returning chunks from files that no longer exist.
    """
    global _bm25_retriever, _bm25_doc_count

    with _bm25_lock:
        _bm25_retriever = None
        _bm25_doc_count = 0
        _bm25_scoped.clear()


def _dedupe(documents: list[Document]) -> list[Document]:
    """Remove duplicate chunks, keeping the first occurrence.

    Keyed on page_content, not id. Vector and keyword search frequently return
    the same chunk, and re-uploading a file stores its text under fresh ids, so
    id misses duplicates that the model would otherwise read twice. Identical
    text is identical waste regardless of which id it carries.
    """
    seen = set()
    unique = []

    for doc in documents:
        key = doc.page_content
        if key in seen:
            continue
        seen.add(key)
        unique.append(doc)

    return unique


def hybrid_search(vector_store, query: str, sources=None) -> list[Document]:
    """Return the merged candidate pool from vector + BM25 keyword search.

    sources narrows BOTH retrievers to the given filenames before ranking. The
    alternative - search everything, then throw away the chunks that came from
    other files - only finds a small file by luck: a resume owns 9 of 241
    chunks, so it has to outvote 232 unrelated ones just to reach the pool.

    Degrades to vector-only if BM25 is unavailable or fails, so a broken keyword
    index never takes down retrieval.
    """
    search_kwargs = {"k": VECTOR_K}
    where = _source_filter(sources)
    if where:
        search_kwargs["filter"] = where

    vector_retriever = vector_store.as_retriever(search_kwargs=search_kwargs)
    vector_docs = vector_retriever.invoke(query)

    try:
        bm25 = get_bm25_retriever(vector_store, sources)
        keyword_docs = bm25.invoke(query) if bm25 else []
    except Exception as exc:
        print(f"[hybrid] keyword search failed, using vector-only: {exc}")
        keyword_docs = []

    merged = _dedupe(vector_docs + keyword_docs)

    print(
        f"[hybrid] vector={len(vector_docs)} keyword={len(keyword_docs)} "
        f"merged_unique={len(merged)}"
        + (f" scoped_to={list(sources)}" if sources else "")
    )

    return merged


def _interleave(pools: list[list[Document]], cap: int) -> list[Document]:
    """Round-robin the pools by rank, deduping, until the cap is reached.

    Concatenating would let the first phrasing fill the whole budget and push
    every later phrasing's best matches past the cap. Walking rank 0 of each
    pool, then rank 1 of each, guarantees each variant contributes its top hits
    before any variant's weaker tail survives.
    """
    seen = set()
    interleaved = []

    deepest = max((len(pool) for pool in pools), default=0)

    for rank in range(deepest):
        for pool in pools:
            if rank >= len(pool):
                continue

            doc = pool[rank]
            if doc.page_content in seen:
                continue

            seen.add(doc.page_content)
            interleaved.append(doc)

            if len(interleaved) >= cap:
                return interleaved

    return interleaved


def multi_query_search(
    vector_store,
    queries: list[str],
    cap: int = MAX_CANDIDATES,
    sources=None,
) -> list[Document]:
    """Run hybrid search for every phrasing and merge one candidate pool.

    The pools are handed to the cross-encoder reranker unscored and unfused:
    retrieval ranks are ordinal and not comparable across phrasings, so ranking
    is left entirely to the reranker downstream.
    """
    pools = [hybrid_search(vector_store, q, sources) for q in queries]
    merged = _interleave(pools, cap)

    print(
        f"[multi-query] {len(queries)} phrasings -> "
        f"{sum(len(p) for p in pools)} raw -> {len(merged)} candidates"
    )

    return merged
