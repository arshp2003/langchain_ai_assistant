"""Manual test for hybrid retrieval (vector search + BM25 keyword search).

Run:  python test_hybrid_search.py

Tests:
  1. Hybrid returns a superset of vector-only results
  2. BM25 contributes chunks that vector search missed
  3. Duplicates across the two retrievers are removed
  4. Failure of keyword search degrades to vector-only
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from langchain_chroma import Chroma
from embeddings import LocalEmbeddings
import hybrid_search
from hybrid_search import hybrid_search as do_hybrid, _dedupe, tokenize, VECTOR_K


def get_store():
    return Chroma(
        collection_name="langchain_notes",
        embedding_function=LocalEmbeddings(),
        persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
    )


store = get_store()

# A term-heavy query: BM25 should shine on exact acronym/keyword matches that
# embedding similarity alone can underrank.
QUERY = "RAG retrieval augmented generation"

# -----------------------------
# 1 + 2: Hybrid is a superset, BM25 adds recall
# -----------------------------
print(f"Query: {QUERY}\n")

vector_only = store.as_retriever(
    search_kwargs={"k": VECTOR_K}
).invoke(QUERY)
merged = do_hybrid(store, QUERY)

vector_texts = {d.page_content for d in vector_only}
merged_texts = {d.page_content for d in merged}

assert vector_texts <= merged_texts, "Hybrid must contain every vector-only chunk"
print(f"Hybrid is a superset of vector-only: OK")

added = merged_texts - vector_texts
print(f"BM25 added {len(added)} chunk(s) vector search missed")
assert len(added) > 0, "BM25 should contribute at least one new chunk"

for i, text in enumerate(sorted(added), 1):
    print(f"  +[{i}] {text[:70].strip()}")

# -----------------------------
# 3: Duplicates removed
# -----------------------------
assert len(merged) == len(merged_texts), "Merged pool must not contain duplicates"
print(f"\nDeduplication across retrievers: OK ({len(merged)} unique)")

# A chunk appearing in both lists must survive exactly once. Built from
# distinct texts so the expected count is unambiguous.
first_seen = {}
for doc in merged:
    first_seen.setdefault(doc.page_content, doc)
distinct = list(first_seen.values())[:3]

probe = distinct + distinct[0:1]
assert len(_dedupe(probe)) == len(distinct), (
    "_dedupe must collapse identical chunks"
)
print("Explicit duplicate collapse: OK")

# -----------------------------
# 4: Keyword failure degrades to vector-only
# -----------------------------
original = hybrid_search.get_bm25_retriever


def boom(_vs, sources=None):
    raise RuntimeError("simulated BM25 outage")


hybrid_search.get_bm25_retriever = boom
try:
    degraded = do_hybrid(store, QUERY)
finally:
    hybrid_search.get_bm25_retriever = original

assert {d.page_content for d in degraded} == vector_texts, (
    "On keyword failure, hybrid must return exactly the vector-only chunks"
)
print("Fallback to vector-only on failure: OK")

# -----------------------------
# 5: Cache rebuilds when corpus changes
# -----------------------------
hybrid_search.invalidate_bm25()
assert hybrid_search._bm25_retriever is None, "invalidate must drop the cached index"

do_hybrid(store, QUERY)
assert hybrid_search._bm25_retriever is not None, "search must rebuild the index"
print("Index invalidation + lazy rebuild: OK")

# -----------------------------
# 6: Scoped search reads only the requested file
# -----------------------------
from collections import Counter

counts = Counter((m or {}).get("source", "") for m in store.get()["metadatas"])
target = "Arsh_GenAI.pdf" if "Arsh_GenAI.pdf" in counts else min(counts, key=counts.get)

scoped = do_hybrid(store, QUERY, sources=[target])
assert scoped, f"scoping to {target} must still find its chunks"
assert {d.metadata.get("source") for d in scoped} == {target}, (
    "A scoped pool must never contain another file's chunks"
)
print(
    f"\nScoped to {target}: {len(scoped)} candidates, "
    f"corpus owns {counts[target]} of {sum(counts.values())} chunks"
)

# The whole point of pushing the filter down instead of applying it afterwards:
# a file holding a handful of chunks has to outvote the rest of the corpus just
# to reach an unscoped top-k pool.
file_specific = "programming language Python skills"
unscoped = do_hybrid(store, file_specific)
reach = sum(1 for d in unscoped if d.metadata.get("source") == target)
print(
    f"Same query unscoped: {reach} of {len(unscoped)} candidates came from "
    f"{target} -> the other {counts[target] - reach} were invisible to a "
    f"post-filter"
)

# -----------------------------
# 7: Tokenizer gives corpus and query a shared vocabulary
# -----------------------------
assert tokenize("MVCC.") == ["mvcc"], tokenize("MVCC.")
assert tokenize("LangChain provides RAG!") == ["langchain", "provides", "rag"]
assert tokenize("MVCC.") != "MVCC.".split(), (
    "The library default keeps case and punctuation, so 'MVCC.' never matched "
    "a query for 'MVCC'"
)
print("Punctuation- and case-insensitive tokenizing: OK")

print("\nPASS: hybrid retrieval verified")
