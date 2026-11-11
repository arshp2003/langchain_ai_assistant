"""Manual test for conditional multi-query expansion.

Run:  python test_conditional_expansion.py

The retrievers, reranker and downstream stages are stubbed so each check asserts
one thing about the gate: expansion (the only stage that costs a model request)
fires on weak retrieval scores and stays silent on strong ones. Zero API calls.
`filter_relevant` is left real, since the honest "nothing found" reply depends
on the same threshold the gate uses.

Follow-up: python test_query_transform.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.documents import Document

import agent_graph
import hybrid_search as hybrid_module
import reranker as reranker_module
import trimmer as trimmer_module
import compressor as compressor_module
import validator as validator_module
from agent_graph import search_documents, _only_from_files

EXPANSION_MARKER = "alternative phrasings"


class FakeResponse:
    def __init__(self, text):
        self.content = text


def D(text, source="notes.pdf"):
    return Document(page_content=text, metadata={"source": source})


def run_search(query, filenames=None):
    func = getattr(search_documents, "func", None) or search_documents
    return func(query, filenames)


class Harness:
    """Drive search_documents with the retrieval stack faked.

    scores_single / scores_multi are the cross-encoder scores returned per pass,
    highest first, which is what rerank() actually guarantees.
    """

    def __init__(self, pool=None, scores_single=(), scores_multi=None):
        self.pool = pool if pool is not None else [D("chunk A"), D("chunk B")]
        self.scores_single = list(scores_single)
        self.scores_multi = (
            self.scores_single if scores_multi is None else list(scores_multi)
        )
        self.single_pass_calls = 0
        self.multi_calls = 0
        self.expansion_requests = 0
        self.phrasings = []
        self._saved = {}

    def hybrid(self, store, query):
        self.single_pass_calls += 1
        return list(self.pool)

    def multi(self, store, queries, cap=30):
        self.multi_calls += 1
        self.phrasings = list(queries)
        return list(self.pool) + [D("chunk C")]

    def rerank(self, query, docs, top_k=3):
        scores = self.scores_multi if self.multi_calls else self.scores_single
        return list(zip(docs, scores))[:top_k]

    def invoke(self, runnable, messages):
        if EXPANSION_MARKER in messages[0].content:
            self.expansion_requests += 1
            return FakeResponse("phrased one way here\nphrased another way here")
        return FakeResponse("chunk A is the answer.")

    def __enter__(self):
        self._saved = {
            (hybrid_module, "hybrid_search"): hybrid_module.hybrid_search,
            (hybrid_module, "multi_query_search"): hybrid_module.multi_query_search,
            (reranker_module, "rerank"): reranker_module.rerank,
            (agent_graph, "invoke_model"): agent_graph.invoke_model,
            (trimmer_module, "trim_passages"): trimmer_module.trim_passages,
            (compressor_module, "compress_context"): compressor_module.compress_context,
            (validator_module, "validate_compression"): validator_module.validate_compression,
        }
        hybrid_module.hybrid_search = self.hybrid
        hybrid_module.multi_query_search = self.multi
        reranker_module.rerank = self.rerank
        agent_graph.invoke_model = self.invoke
        trimmer_module.trim_passages = lambda q, docs: docs
        compressor_module.compress_context = lambda q, docs: docs
        validator_module.validate_compression = lambda q, trimmed, docs: docs
        return self

    def __exit__(self, *exc):
        for (module, name), original in self._saved.items():
            setattr(module, name, original)
        return False


def run(harness, query, filenames=None):
    with harness:
        return harness, run_search(query, filenames)


# -----------------------------
# 1. Strong retrieval pays nothing
# -----------------------------
h, out = run(Harness(scores_single=[5.9, 2.1]), "how does chunking work")
assert h.multi_calls == 0, "strong first pass must not expand"
assert h.phrasings == [], "expand_queries must not even run"
assert h.expansion_requests == 0, "expansion costs a request; none allowed here"
assert "chunk A" in out, out
print("1. Strong score (+5.9) -> single pass, 0 model requests: OK")

# -----------------------------
# 2. Weak retrieval expands exactly once
# -----------------------------
h, out = run(Harness(scores_single=[-4.7, -6.0]), "PostgreSQL MVCC concurrency control")
assert h.multi_calls == 1, h.multi_calls
assert h.expansion_requests == 1, h.expansion_requests
assert h.single_pass_calls == 1, "original pool not re-searched as a phrasing"
assert h.phrasings[0] == "PostgreSQL MVCC concurrency control", h.phrasings
assert len(h.phrasings) == 3, h.phrasings
assert out == "No relevant documents found in the knowledge base.", out
print("2. Weak score (-4.7) -> expands once, then reports honestly: OK")

# -----------------------------
# 3. Empty candidate pool still expands
# -----------------------------
h, out = run(Harness(pool=[], scores_single=[]), "quantum tunnelling in semiconductors")
assert h.multi_calls == 1, h.multi_calls
assert out == "No relevant documents found in the knowledge base.", out
print("3. Empty candidate pool -> expands: OK")

# -----------------------------
# 4. Score exactly at the bar counts as found
# -----------------------------
h, out = run(Harness(scores_single=[1.0]), "langchain text splitters")
assert h.multi_calls == 0, h.multi_calls
assert "chunk A" in out, out
print("4. Score == DEFAULT_MIN_SCORE is relevant, no expansion: OK")

# -----------------------------
# 5. Expansion recovers the answer
# -----------------------------
h, out = run(Harness(scores_single=[-3.0], scores_multi=[7.4, 5.1, 2.0]),
             "what is retrieval augmented generation")
assert h.multi_calls == 1, h.multi_calls
assert h.expansion_requests == 1, h.expansion_requests
assert "chunk A" in out, out
print("5. Expanded pool surfaces the answer, one request total: OK")

# -----------------------------
# 6. Filename filter
# -----------------------------
docs = [D("a", "one.pdf"), D("b", "two.pdf"), D("c", "missing.pdf")]
assert len(_only_from_files(docs, None)) == 3, "no filter passes everything through"
assert [d.page_content for d in _only_from_files(docs, ["one.pdf", "two.pdf"])] == ["a", "b"]
assert _only_from_files(docs, ["nope.pdf"]) == []
print("6. _only_from_files keeps only the requested sources: OK")

# -----------------------------
# 7. Attachments requested, none matched -> names the files
# -----------------------------
h, out = run(Harness(pool=[D("a", "other.pdf")], scores_single=[0.3]),
             "anything", filenames=["wanted.pdf"])
assert "wanted.pdf" in out, out
print("7. Attached files with no matching chunk -> names them: OK")

# -----------------------------
# 8. Same gate against the real corpus (real Chroma + real cross-encoder)
#    Only invoke_model is stubbed, so this still costs zero API requests.
# -----------------------------
from embeddings import LocalEmbeddings
from langchain_chroma import Chroma

real_invoke = agent_graph.invoke_model
seen = []


def counting_invoke(runnable, messages):
    if EXPANSION_MARKER in str(messages[0].content if hasattr(messages[0], "content") else messages):
        seen.append("expansion")
    return FakeResponse("compressed summary of the retrieved context")


agent_graph.invoke_model = counting_invoke
store = Chroma(
    collection_name="langchain_notes",
    embedding_function=LocalEmbeddings(),
    persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db"),
)
print(f"\n8. Real corpus ({store._collection.count()} chunks):")
try:
    for label, q in [
        ("covered", "text splitting chunk size strategy"),
        ("not covered", "PostgreSQL MVCC concurrency control"),
    ]:
        seen.clear()
        out = run_search(q)
        print(
            f"   {label:<12} expansion requests={len(seen)} "
            f"found={'yes' if 'Document' in out else 'no'}"
        )
finally:
    agent_graph.invoke_model = real_invoke

print("\nPASS: expansion is gated on retrieval quality")
