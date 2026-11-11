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
"""

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


def _load_all_documents(vector_store) -> list[Document]:
    """Pull every chunk out of the Chroma collection as LangChain Documents."""
    raw = vector_store.get()

    documents = []
    for doc_id, text, metadata in zip(
        raw["ids"], raw["documents"], raw["metadatas"]
    ):
        documents.append(
            Document(page_content=text, metadata=metadata or {}, id=doc_id)
        )
    return documents


def _build_bm25(vector_store):
    """Build a BM25Retriever over the persisted chunks."""
    from langchain_community.retrievers import BM25Retriever

    documents = _load_all_documents(vector_store)
    if not documents:
        return None

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = KEYWORD_K
    return retriever


def get_bm25_retriever(vector_store):
    """Return the cached BM25 retriever, rebuilding it if the corpus changed."""
    global _bm25_retriever, _bm25_doc_count

    with _bm25_lock:
        current_count = vector_store._collection.count()

        if _bm25_retriever is None or _bm25_doc_count != current_count:
            _bm25_retriever = _build_bm25(vector_store)
            _bm25_doc_count = current_count
            print(f"[hybrid] BM25 index built over {current_count} chunks")

        return _bm25_retriever


def invalidate_bm25() -> None:
    """Drop the cached index so the next search rebuilds it.

    Call after uploading or deleting a document, otherwise keyword search keeps
    returning chunks from files that no longer exist.
    """
    global _bm25_retriever, _bm25_doc_count

    with _bm25_lock:
        _bm25_retriever = None
        _bm25_doc_count = 0


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


def hybrid_search(vector_store, query: str) -> list[Document]:
    """Return the merged candidate pool from vector + BM25 keyword search.

    Degrades to vector-only if BM25 is unavailable or fails, so a broken keyword
    index never takes down retrieval.
    """
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": VECTOR_K}
    )
    vector_docs = vector_retriever.invoke(query)

    try:
        bm25 = get_bm25_retriever(vector_store)
        keyword_docs = bm25.invoke(query) if bm25 else []
    except Exception as exc:
        print(f"[hybrid] keyword search failed, using vector-only: {exc}")
        keyword_docs = []

    merged = _dedupe(vector_docs + keyword_docs)

    print(
        f"[hybrid] vector={len(vector_docs)} keyword={len(keyword_docs)} "
        f"merged_unique={len(merged)}"
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
) -> list[Document]:
    """Run hybrid search for every phrasing and merge one candidate pool.

    The pools are handed to the cross-encoder reranker unscored and unfused:
    retrieval ranks are ordinal and not comparable across phrasings, so ranking
    is left entirely to the reranker downstream.
    """
    pools = [hybrid_search(vector_store, q) for q in queries]
    merged = _interleave(pools, cap)

    print(
        f"[multi-query] {len(queries)} phrasings -> "
        f"{sum(len(p) for p in pools)} raw -> {len(merged)} candidates"
    )

    return merged
