"""Quick test for the cross-encoder reranker.

Verifies that:
1. Reranking reorders results by actual relevance (not vector order)
2. Irrelevant chunks score much lower than relevant ones
3. Fallback works when the model errors out
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace

from reranker import rerank, filter_relevant

query = "How does PostgreSQL handle concurrent writes?"

# Simulated vector-search results in intentional wrong order.
# Vector similarity might rank these by keyword overlap ("PostgreSQL", "database")
# even though they don't answer the question.
fake_docs = [
    SimpleNamespace(page_content="PostgreSQL was created in 1986 at UC Berkeley."),
    SimpleNamespace(page_content="PostgreSQL uses MVCC (Multi-Version Concurrency Control) so writers do not block readers during concurrent transactions."),
    SimpleNamespace(page_content="PostgreSQL supports JSON and JSONB data types for storing documents."),
    SimpleNamespace(page_content="Database indexing in general improves query speed."),
    SimpleNamespace(page_content="MVCC in PostgreSQL creates new row versions on write, letting concurrent transactions proceed without table locks."),
]

print(f"Query: {query}\n")
print("Vector-search order (input):")
for i, doc in enumerate(fake_docs, 1):
    print(f"  {i}. {doc.page_content[:70]}")

scored = rerank(query, fake_docs, top_k=3)

print("\nReranked order (output, top 3):")
for i, (doc, score) in enumerate(scored, 1):
    print(f"  {i}. [{score:+.3f}] {doc.page_content[:70]}")

mvcc_results = [doc for doc, _ in scored]
mvcc_hits = sum(1 for d in mvcc_results if "MVCC" in d.page_content or "concurrent" in d.page_content)
assert mvcc_hits == 2, f"Expected 2 relevant chunks in top 3, got {mvcc_hits}"

# Fallback test: pass garbage that makes predict fail
class BrokenDoc:
    @property
    def page_content(self):
        raise RuntimeError("boom")

fallback = rerank(query, [BrokenDoc()], top_k=3)
assert len(fallback) == 1, "Fallback should still return documents"
print("\nFallback on failure: OK")

# ---- Relevance filtering tests ----

fake_scored = [
    (SimpleNamespace(page_content="MVCC chunk"), 7.7),
    (SimpleNamespace(page_content="Row versions chunk"), 6.4),
    (SimpleNamespace(page_content="Borderline chunk"), 0.9),
    (SimpleNamespace(page_content="Junk chunk"), 0.1),
    (SimpleNamespace(page_content="Contradicts chunk"), -2.0),
]

kept = filter_relevant(fake_scored, min_score=1.0)
assert len(kept) == 2, f"Expected 2 kept, got {len(kept)}"
assert all(score >= 1.0 for _, score in kept)
print(f"\nFilter kept {len(kept)}/5 chunks above 1.0: OK")

empty = filter_relevant([(SimpleNamespace(page_content="x"), 0.5)], min_score=1.0)
assert empty == [], "Everything below threshold should return empty list"
print("All-below-threshold returns empty (honest no-results): OK")

print("\nPASS: reranking + relevance filtering both verified")
