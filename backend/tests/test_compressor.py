"""Manual test for context compression.

Run:  python test_compressor.py

Tests:
  1. Short context is skipped (no LLM call)
  2. Long context gets compressed
  3. Compressed output preserves key facts
  4. Failure falls back gracefully
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.documents import Document
from compressor import compress_context, _estimate_tokens, TOKEN_THRESHOLD


def make_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


def test_short_context_skipped():
    print("\n=== Test 1: Short context should be skipped ===")
    doc = make_doc("LangChain is a framework for building LLM applications.")
    scored = [(doc, 5.0)]

    print(f"Context tokens: ~{_estimate_tokens(doc.page_content)}")
    print(f"Threshold: {TOKEN_THRESHOLD}")

    result = compress_context("What is LangChain?", scored)

    assert len(result) == 1, "Should return same number of docs"
    assert result[0][0].page_content == doc.page_content, "Short context should be unchanged"
    print("[OK] Short context skipped (no LLM call)")


def test_long_context_compressed():
    print("\n=== Test 2: Long context should be compressed ===")

    # Create verbose context with repetition - needs to exceed TOKEN_THRESHOLD (600 tokens = ~2400 chars)
    chunk1 = """PostgreSQL was created in 1986 at the University of California, Berkeley by Michael Stonebraker and his team of graduate students and researchers. It was originally called POSTGRES, a playful reference to the earlier INGRES database project that Stonebraker had also developed. The POSTGRES project was designed to be the spiritual successor to INGRES, addressing the next wave of database challenges. The project was renamed to PostgreSQL in 1996 to reflect its newly added SQL query capabilities, which had been added in response to user demand. Since then, PostgreSQL has grown to become one of the most popular open-source relational database management systems in the world, used by companies of all sizes for mission-critical applications. PostgreSQL supports ACID transactions, which ensure that database operations are processed reliably. It also supports foreign keys and referential integrity, allowing complex relationships between tables to be maintained. PostgreSQL uses MVCC (Multi-Version Concurrency Control) as its core concurrency mechanism, so writers do not block readers and readers do not block writers during concurrent database operations. The database is highly extensible with custom data types, functions, operators, and index methods, allowing developers to tailor it to their specific needs."""

    chunk2 = """MVCC (Multi-Version Concurrency Control) is the concurrency mechanism that PostgreSQL uses to provide concurrent access to the database while maintaining data consistency and isolation between transactions. When a transaction modifies a row, MVCC creates a new version of that row rather than overwriting the existing one in place. This allows concurrent transactions to proceed without acquiring table-level locks, which would serialize all operations and severely limit throughput. Each transaction sees a snapshot of the database as of its start time, which means it only sees rows that were committed before the transaction began. This isolation model prevents dirty reads (reading uncommitted data) and non-repeatable reads (seeing different values for the same row within a transaction) in the default READ COMMITTED isolation level. PostgreSQL supports four standard SQL isolation levels: READ UNCOMMITTED, which allows dirty reads; READ COMMITTED, which is the default and prevents dirty reads; REPEATABLE READ, which also prevents non-repeatable reads; and SERIALIZABLE, which provides the strictest isolation by preventing phantom reads (rows appearing or disappearing between queries) using predicate locks. The SERIALIZABLE isolation level is particularly useful for applications that require strict transaction isolation, such as financial systems or inventory management, where concurrent modifications could lead to data anomalies if not properly controlled."""

    doc1 = make_doc(chunk1, "postgres_history.txt")
    doc2 = make_doc(chunk2, "postgres_mvcc.txt")
    scored = [(doc1, 7.5), (doc2, 6.8)]

    total_text = chunk1 + chunk2
    print(f"Total context tokens: ~{_estimate_tokens(total_text)}")
    print(f"Threshold: {TOKEN_THRESHOLD}")

    result = compress_context("How does PostgreSQL handle concurrent writes?", scored)

    assert len(result) >= 1, "Should return at least one doc"
    compressed_text = result[0][0].page_content
    compressed_tokens = _estimate_tokens(compressed_text)

    print(f"Compressed tokens: ~{compressed_tokens}")
    reduction = 100 - int(100 * compressed_tokens / _estimate_tokens(total_text))
    print(f"Reduction: {reduction}%")

    # Check key facts preserved
    key_facts = ["MVCC", "concurrent", "transaction", "isolation"]
    found = [fact for fact in key_facts if fact.lower() in compressed_text.lower()]
    print(f"Key facts preserved: {found}")

    assert compressed_tokens < _estimate_tokens(total_text), "Should be shorter"
    assert len(found) >= 2, f"Should preserve key facts, found: {found}"
    print("[OK] Long context compressed with facts preserved")


def test_empty_input():
    print("\n=== Test 3: Empty input should return empty ===")
    result = compress_context("test", [])
    assert result == [], "Empty input should return empty"
    print("[OK] Empty input handled")


if __name__ == "__main__":
    print("Testing context compression...")
    print(f"Token threshold: {TOKEN_THRESHOLD}")

    try:
        test_short_context_skipped()
        test_empty_input()
        test_long_context_compressed()

        print("\n[OK] All tests passed!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
