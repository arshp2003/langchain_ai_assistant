"""Manual test for context validation.

Run:  python test_validator.py

Tests:
  1. Compressed context with high similarity is kept
  2. Compressed context with low similarity falls back to trimmed
  3. Empty input returns trimmed docs
  4. Exception falls back gracefully
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_core.documents import Document
from validator import validate_compression, _cosine_similarity


def make_doc(content: str, source: str = "test.pdf") -> Document:
    return Document(page_content=content, metadata={"source": source})


def test_cosine_similarity():
    print("\n=== Test 1: Cosine similarity function ===")

    # Identical vectors
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    sim = _cosine_similarity(a, b)
    print(f"Identical vectors: {sim:.3f}")
    assert abs(sim - 1.0) < 0.001, "Identical vectors should have similarity 1.0"

    # Orthogonal vectors
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    sim = _cosine_similarity(a, b)
    print(f"Orthogonal vectors: {sim:.3f}")
    assert abs(sim - 0.0) < 0.001, "Orthogonal vectors should have similarity 0.0"

    # Opposite vectors
    a = [1.0, 0.0, 0.0]
    b = [-1.0, 0.0, 0.0]
    sim = _cosine_similarity(a, b)
    print(f"Opposite vectors: {sim:.3f}")
    assert abs(sim - (-1.0)) < 0.001, "Opposite vectors should have similarity -1.0"

    print("[OK] Cosine similarity function works")


def test_validation_keeps_good_compression():
    print("\n=== Test 2: Good compression should be kept ===")

    query = "What is MVCC in PostgreSQL?"

    # Trimmed context (before compression)
    trimmed = [
        (make_doc("MVCC is Multi-Version Concurrency Control. It allows concurrent transactions without locks.", "chunk1.txt"), 7.5),
    ]

    # Compressed context (after compression - still contains key info)
    compressed = [
        (make_doc("MVCC (Multi-Version Concurrency Control) enables concurrent transactions without locks.", "chunk1.txt"), 7.5),
    ]

    result = validate_compression(query, trimmed, compressed)

    # Should return compressed (same length, high similarity)
    assert len(result) == 1, "Should return one doc"
    assert result[0][0].page_content == compressed[0][0].page_content, "Should keep compressed version"
    print("[OK] Good compression kept")


def test_validation_falls_back_on_bad_compression():
    print("\n=== Test 3: Bad compression should fall back to trimmed ===")

    query = "What is MVCC in PostgreSQL and how does it work?"

    # Trimmed context (before compression)
    trimmed = [
        (make_doc("MVCC is Multi-Version Concurrency Control. It creates new row versions on write. Each transaction sees a snapshot. This prevents dirty reads and allows concurrent access.", "chunk1.txt"), 7.5),
    ]

    # Badly compressed context (lost key information)
    compressed = [
        (make_doc("PostgreSQL is a database.", "chunk1.txt"), 7.5),
    ]

    result = validate_compression(query, trimmed, compressed)

    # Should return trimmed (compressed lost too much info)
    assert len(result) == 1, "Should return one doc"
    assert result[0][0].page_content == trimmed[0][0].page_content, "Should fall back to trimmed version"
    print("[OK] Bad compression fell back to trimmed")


def test_empty_input():
    print("\n=== Test 4: Empty input should return trimmed ===")

    query = "test query"
    trimmed = []
    compressed = []

    result = validate_compression(query, trimmed, compressed)
    assert result == [], "Empty input should return empty"
    print("[OK] Empty input handled")


def test_exception_fallback():
    print("\n=== Test 5: Exception should fall back to trimmed ===")

    query = "test query"
    trimmed = [(make_doc("some content"), 5.0)]
    compressed = [(make_doc("compressed content"), 5.0)]

    # Mock embeddings to raise an exception
    import validator
    original_embed = validator.embeddings.embed_query

    def mock_embed(text):
        raise RuntimeError("Embedding failed")

    validator.embeddings.embed_query = mock_embed

    try:
        result = validate_compression(query, trimmed, compressed)
        assert result == trimmed, "Should fall back to trimmed on exception"
        print("[OK] Exception fell back to trimmed")
    finally:
        validator.embeddings.embed_query = original_embed


if __name__ == "__main__":
    print("Testing context validation...")

    try:
        test_cosine_similarity()
        test_empty_input()
        test_validation_keeps_good_compression()
        test_validation_falls_back_on_bad_compression()
        test_exception_fallback()

        print("\n" + "=" * 50)
        print("[OK] All tests passed!")
        print("=" * 50)
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
