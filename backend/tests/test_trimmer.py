"""Tests for passage trimming (relative-margin strategy).

Verifies:
1. Relevant sentences are kept, filler is dropped
2. Kept sentences stay in original document order (not score order)
3. Floor rule: best sentence always kept (automatic with relative margin)
4. Single-sentence chunks pass through unchanged
5. Failure in sentence scoring keeps chunks as-is (graceful degradation)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from types import SimpleNamespace

from trimmer import trim_passages, split_sentences

query = "How does PostgreSQL handle concurrent writes?"

# Chunk with two near-equally-relevant sentences (within margin of each other)
# sandwiched between filler, to prove order is restored by original position.
filler_chunk = SimpleNamespace(page_content=(
    "PostgreSQL was created in 1986 at UC Berkeley as an open-source project. "
    "It uses MVCC (Multi-Version Concurrency Control) so writers do not block "
    "readers during concurrent transactions. "
    "The project was originally named POSTGRES. "
    "MVCC creates new row versions on write, allowing concurrent transactions "
    "to proceed without acquiring table locks."
))

# Sentence splitting sanity
sentences = split_sentences(filler_chunk.page_content)
assert len(sentences) == 4, f"Expected 4 sentences, got {len(sentences)}: {sentences}"
print(f"Split into {len(sentences)} sentences: OK")

trimmed = trim_passages(query, [(filler_chunk, 7.7)])

assert len(trimmed) == 1, "Should return one (doc, score) tuple"
trimmed_doc, chunk_score = trimmed[0]
assert chunk_score == 7.7, "Chunk score must pass through unchanged"
trimmed_text = trimmed_doc.page_content
assert "MVCC" in trimmed_text, "Relevant sentence must be kept"
assert "1986" not in trimmed_text, "History filler sentence should be dropped"
assert "POSTGRES" not in trimmed_text, "Naming filler sentence should be dropped"
assert len(trimmed_text) < len(filler_chunk.page_content), "Trimming must shorten the text"

order_kept = len(split_sentences(trimmed_text))
if order_kept == 2:
    i_first = trimmed_text.index("MVCC (Multi-Version")
    i_second = trimmed_text.index("MVCC creates new row versions")
    assert i_first < i_second, "Kept sentences must preserve original order"
print(f"Trimmed text ({len(trimmed_text)} chars, was {len(filler_chunk.page_content)}), "
      f"{order_kept} sentence(s) kept:")
print(f"  {trimmed_text}")

# Floor rule: chunk where NO sentence is relevant must still keep its
# single best sentence, never become empty.
boring_chunk = SimpleNamespace(page_content=(
    "PostgreSQL was created in 1986 at UC Berkeley. "
    "The project was led by Michael Stonebraker."
))
floored = trim_passages(query, [(boring_chunk, 0.2)])
floored_text = floored[0][0].page_content
assert floored_text, "Floor rule: best sentence must be kept"
assert len(split_sentences(floored_text)) == 1, "Floor keeps exactly one sentence"
print(f"Floor rule (nothing relevant, kept best): {floored_text}")

# Single-sentence chunk passes through untouched
one_liner = SimpleNamespace(page_content="PostgreSQL uses MVCC for concurrency.")
passthrough = trim_passages(query, [(one_liner, 5.0)])
assert passthrough[0][0].page_content == one_liner.page_content, "1-sentence chunk unchanged"
print("Single-sentence chunk passthrough: OK")

# Graceful degradation: scoring failure keeps original chunks
class BrokenDoc:
    @property
    def page_content(self):
        raise RuntimeError("boom")

degraded = trim_passages(query, [(BrokenDoc(), 3.0)])
assert len(degraded) == 1, "On failure, chunks pass through unchanged"
print("Scoring failure passthrough: OK")

# Empty input
assert trim_passages(query, []) == []
print("Empty input: OK")

print("\nPASS: passage trimming verified")
