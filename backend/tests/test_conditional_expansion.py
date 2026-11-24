"""Manual test for conditional multi-query expansion and file scoping.

Run:  python test_conditional_expansion.py

The retrievers, reranker and downstream stages are stubbed so each check asserts
one thing about the gate: expansion (the only stage that costs a model request)
fires when the candidate pool is EMPTY and stays silent otherwise. Zero API
calls. `filter_relevant` is left real, since the LOW CONFIDENCE reply and the
honest "nothing found" reply both depend on the same threshold it uses.

Why empty-pool and not low-score: every phrasing is fed to the same two
retrievers that just ranked the weak pool, so rewording cannot repair a ranking
problem - it can only find chunks the first wording could not reach at all.

Follow-up: python test_query_transform.py
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

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

    pool is what the first search pass finds; expanded_pool is what the pass
    over generated phrasings finds (defaults to the first pool plus one chunk
    only the rewording reaches). scores_single / scores_multi are the
    cross-encoder scores returned per pass, highest first, which is what
    rerank() actually guarantees.
    """

    def __init__(self, pool=None, scores_single=(), scores_multi=None,
                 expanded_pool=None):
        self.pool = pool if pool is not None else [D("chunk A"), D("chunk B")]
        self.expanded_pool = (
            expanded_pool if expanded_pool is not None
            else self.pool + [D("chunk C")]
        )
        self.scores_single = list(scores_single)
        self.scores_multi = (
            self.scores_single if scores_multi is None else list(scores_multi)
        )
        self.multi_calls = 0
        self.expansion_requests = 0
        self.phrasings = []
        self.scopes = []
        self._saved = {}

    def multi(self, store, queries, cap=30, sources=None):
        self.multi_calls += 1
        self.phrasings.append(list(queries))
        self.scopes.append(sorted(sources) if sources else None)
        if self.multi_calls == 1:
            return list(self.pool)
        return list(self.expanded_pool)

    def rerank(self, query, docs, top_k=3):
        scores = self.scores_multi if self.multi_calls > 1 else self.scores_single
        return list(zip(docs, scores))[:top_k]

    def invoke(self, runnable, messages):
        if EXPANSION_MARKER in messages[0].content:
            self.expansion_requests += 1
            return FakeResponse("phrased one way here\nphrased another way here")
        return FakeResponse("chunk A is the answer.")

    def __enter__(self):
        self._saved = {
            (hybrid_module, "multi_query_search"): hybrid_module.multi_query_search,
            (reranker_module, "rerank"): reranker_module.rerank,
            (agent_graph, "invoke_model"): agent_graph.invoke_model,
            (trimmer_module, "trim_passages"): trimmer_module.trim_passages,
            (compressor_module, "compress_context"): compressor_module.compress_context,
            (validator_module, "validate_compression"): validator_module.validate_compression,
        }
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
assert h.multi_calls == 1, "the first pass must search, not skip"
assert h.expansion_requests == 0, "expansion costs a request; none allowed here"
assert len(h.phrasings) == 1, f"expected one search pass, got {h.phrasings}"
assert "chunk A" in out, out
print("1. Strong score (+5.9) -> one pass, 0 model requests: OK")

# -----------------------------
# 2. Weak scores do NOT trigger expansion
#    This is the whole point of the gate: the pool is full, the ranking is just
#    poor, and rewording goes back through the same retrievers. The model gets
#    the best candidate labelled LOW CONFIDENCE instead.
# -----------------------------
h, out = run(Harness(scores_single=[-4.7, -6.0]), "PostgreSQL MVCC concurrency control")
assert h.expansion_requests == 0, h.expansion_requests
assert len(h.phrasings) == 1, f"weak-but-nonempty pool must not expand: {h.phrasings}"
assert "LOW CONFIDENCE" in out, out
assert "-4.70" in out, f"the score itself must reach the model: {out}"
assert "chunk A" in out, out
print("2. Weak score (-4.7) -> no request spent, best match returned flagged: OK")

# -----------------------------
# 3. Empty candidate pool is the one trigger
# -----------------------------
h, out = run(Harness(pool=[], expanded_pool=[], scores_single=[], scores_multi=[]),
             "quantum tunnelling in semiconductors")
assert h.expansion_requests == 1, h.expansion_requests
assert len(h.phrasings) == 2, h.phrasings
assert h.phrasings[0] == ["quantum tunnelling in semiconductors"], h.phrasings
assert len(h.phrasings[1]) == 3, "expansion contributes its own phrasings"
assert out == "No relevant documents found in the knowledge base.", out
print("3. Empty candidate pool -> expands exactly once: OK")

# -----------------------------
# 4. Score exactly at the bar counts as found
# -----------------------------
h, out = run(Harness(scores_single=[1.0]), "langchain text splitters")
assert h.expansion_requests == 0, h.expansion_requests
assert "chunk A" in out, out
print("4. Score == DEFAULT_MIN_SCORE is relevant, no expansion: OK")

# -----------------------------
# 5. Expansion recovers the answer
# -----------------------------
h, out = run(Harness(pool=[], scores_single=[], scores_multi=[7.4, 5.1, 2.0]),
             "what is retrieval augmented generation")
assert h.expansion_requests == 1, h.expansion_requests
assert "chunk C" in out, out
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
# 7. Filenames reach the retrievers as a scope, not just a post-filter
# -----------------------------
h, out = run(Harness(pool=[D("a", "wanted.pdf")], scores_single=[4.0]),
             "anything", filenames=["wanted.pdf", "also.pdf"])
assert h.scopes == [["also.pdf", "wanted.pdf"]], h.scopes
assert h.expansion_requests == 0, h.expansion_requests
assert out == "[Document 1 - wanted.pdf]\na", out
print("7. Attached files are pushed down into retrieval as a scope: OK")

# -----------------------------
# 8. Attachments requested, nothing in them at all -> names the files,
#    and the empty scope is what paid for the one expansion request.
# -----------------------------
h, out = run(Harness(pool=[D("a", "other.pdf")], scores_single=[0.3]),
             "anything", filenames=["wanted.pdf"])
assert "wanted.pdf" in out, out
assert h.expansion_requests == 1, "an empty scope is exactly when rewording helps"
assert h.scopes == [["wanted.pdf"], ["wanted.pdf"]], h.scopes
print("8. Attached files with no matching chunk -> names them: OK")

# -----------------------------
# 9. Same gate against the real corpus (real Chroma + real cross-encoder)
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
print(f"\n9. Real corpus ({store._collection.count()} chunks):")
try:
    for label, q, files in [
        ("covered", "text splitting chunk size strategy", None),
        ("not covered", "PostgreSQL MVCC concurrency control", None),
        ("scoped to a file", "programming language", ["Arsh_GenAI.pdf"]),
        ("scoped, wrong file", "programming language", ["does_not_exist.pdf"]),
    ]:
        seen.clear()
        out = run_search(q, files)
        print(
            f"   {label:<18} expansion requests={len(seen)} "
            f"found={'yes' if 'Document' in out else 'no'} "
            f"lowconf={'LOW CONFIDENCE' in out}"
        )
finally:
    agent_graph.invoke_model = real_invoke

print("\nPASS: expansion is gated on an empty candidate pool")
