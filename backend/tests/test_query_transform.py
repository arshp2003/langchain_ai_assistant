"""Manual test for multi-query expansion wiring.

Run:  python test_query_transform.py

The LLM call is stubbed, so this test costs zero API requests. It verifies the
deterministic parts: parsing, interleaving, dedupe, caps, and the fallbacks that
keep retrieval alive when expansion fails.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

from langchain_core.documents import Document
import query_transform
from query_transform import _parse_variants, expand_queries
from hybrid_search import _interleave, multi_query_search


def D(text):
    return Document(page_content=text, metadata={"source": "test.pdf"})


# -----------------------------
# 1. Parsing
# -----------------------------
raw = """1. MVCC concurrency control
- parallel transaction handling
* write-ahead logging WAL
2) Isolation levels in PostgreSQL

xx
"""
variants = _parse_variants(raw)
assert len(variants) == 4, variants
assert not any(v[0].isdigit() for v in variants), "list markers must be stripped"
assert "xx" not in variants, "sub-4-char artifacts dropped"
print("1. Parser strips numbering/bullets, drops short lines: OK")

# -----------------------------
# 2. Interleaving beats concatenation
# -----------------------------
pool_a = [D("a1"), D("a2"), D("a3")]
pool_b = [D("b1"), D("b2")]

order = [d.page_content for d in _interleave([pool_a, pool_b], cap=10)]
assert order == ["a1", "b1", "a2", "b2", "a3"], order
print("2. Round-robin by rank, not concatenated: OK", order)

# -----------------------------
# 3. Cap never evicts a variant's best hit
# -----------------------------
capped = [d.page_content for d in _interleave([pool_a, pool_b], cap=3)]
assert "a1" in capped and "b1" in capped, capped
print("3. Cap keeps each variant's top hit: OK", capped)

# -----------------------------
# 4. Cross-variant dedupe
# -----------------------------
dupes = _interleave([[D("x"), D("y")], [D("x"), D("z")]], cap=10)
assert [d.page_content for d in dupes] == ["x", "y", "z"], dupes
print("4. Same chunk found by two phrasings collapses: OK")

# -----------------------------
# 5. Ragged and empty pools
# -----------------------------
assert _interleave([[], [D("only")]], cap=5)[0].page_content == "only"
assert _interleave([], cap=5) == []
print("5. Ragged / empty pools handled: OK")

# -----------------------------
# 6. Expansion: original query always leads and survives
# -----------------------------
class FakeResponse:
    def __init__(self, text):
        self.content = text


def fake_invoke(runnable, messages):
    return FakeResponse("variant one alpha\nvariant two beta\nvariant three gamma")


# expand_queries imports invoke_model from agent_graph at call time, so
# agent_graph is the only place that needs patching.
import agent_graph
real_invoke = agent_graph.invoke_model
agent_graph.invoke_model = fake_invoke
try:
    queries = expand_queries("original question here", n_variants=3)
finally:
    agent_graph.invoke_model = real_invoke

assert queries[0] == "original question here", "original must lead the list"
assert len(queries) == 4, queries
print("6. Original leads, variants appended: OK", queries)

# -----------------------------
# 7. Expansion failure degrades to original-only
# -----------------------------
def boom(runnable, messages):
    raise RuntimeError("simulated 429 on expansion")


agent_graph.invoke_model = boom
try:
    fallback = expand_queries("original question here")
finally:
    agent_graph.invoke_model = real_invoke

assert fallback == ["original question here"], fallback
print("7. Expansion failure -> original query only: OK")

# -----------------------------
# 8. Duplicate phrasing from the model is dropped
# -----------------------------
agent_graph.invoke_model = fake_invoke
try:
    # Fake response repeats the original wording; it must not appear twice.
    dup_text = "original question here\nvariant one alpha"
    agent_graph.invoke_model = lambda r, m: FakeResponse(dup_text)
    queries = expand_queries("original question here", n_variants=3)
finally:
    agent_graph.invoke_model = real_invoke

assert queries.count("original question here") == 1, queries
print("8. Model echoing the original is deduped: OK", queries)

# -----------------------------
# 9. multi_query_search merges across phrasings against real Chroma
# -----------------------------
from langchain_chroma import Chroma
from embeddings import LocalEmbeddings
from hybrid_search import invalidate_bm25

store = Chroma(
    collection_name="langchain_notes",
    embedding_function=LocalEmbeddings(),
    persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
)

invalidate_bm25()
single = multi_query_search(store, ["RAG retrieval augmented generation"])
multi = multi_query_search(
    store,
    ["RAG retrieval augmented generation", "chunking text splitting strategy"],
)

assert len(single) <= 30, single
assert len(multi) >= len(single), "extra phrasing must not shrink the pool"
assert len(multi) == len({d.page_content for d in multi}), "no duplicates"
print(f"9. Pools merge and stay capped: single={len(single)} multi={len(multi)} OK")

print("\nPASS: multi-query expansion verified")
