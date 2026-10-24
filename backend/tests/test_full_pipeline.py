"""Full pipeline test: vector search → rerank → filter → trim → compress → validate.

Run:  python test_full_pipeline.py

Tests the complete RAG pipeline end-to-end using the existing chroma_db.
"""

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from langchain_chroma import Chroma
from embeddings import LocalEmbeddings
from reranker import rerank, filter_relevant
from trimmer import trim_passages
from compressor import compress_context
from validator import validate_compression


def run_pipeline(query: str) -> dict:
    """Run the full 5-step RAG pipeline and return metrics."""
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    # Load vector store
    embedding_function = LocalEmbeddings()
    vector_store = Chroma(
        collection_name="langchain_notes",
        embedding_function=embedding_function,
        persist_directory=str(Path(__file__).resolve().parent.parent / "chroma_db")
    )

    # Step 0: Vector search
    print("\n[Step 0] Vector search (k=10)...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)
    print(f"  Retrieved {len(docs)} chunks")
    if not docs:
        return {"error": "No documents found"}

    # Step 1: Rerank
    print("\n[Step 1] Cross-encoder reranking...")
    t0 = time.time()
    scored_docs = rerank(query, docs, top_k=3)
    t1 = time.time()
    print(f"  Reranked {len(scored_docs)} chunks in {t1-t0:.2f}s")
    for i, (doc, score) in enumerate(scored_docs, 1):
        print(f"    [{i}] score={score:.2f}, source={doc.metadata.get('source', 'unknown')}")

    # Step 2: Filter
    print("\n[Step 2] Relevance filtering (threshold=1.0)...")
    before_filter = len(scored_docs)
    scored_docs = filter_relevant(scored_docs)
    print(f"  Kept {len(scored_docs)}/{before_filter} chunks")
    if not scored_docs:
        return {"error": "No relevant chunks after filtering"}

    # Step 3: Trim
    print("\n[Step 3] Passage trimming...")
    t0 = time.time()
    trimmed_docs = trim_passages(query, scored_docs)
    t1 = time.time()
    print(f"  Trimmed {len(trimmed_docs)} chunks in {t1-t0:.2f}s")
    for i, (doc, score) in enumerate(trimmed_docs, 1):
        chars = len(doc.page_content)
        print(f"    [{i}] {chars} chars, source={doc.metadata.get('source', 'unknown')}")

    # Step 4: Compress
    print("\n[Step 4] Context compression...")
    t0 = time.time()
    compressed_docs = compress_context(query, trimmed_docs)
    t1 = time.time()
    print(f"  Compressed in {t1-t0:.2f}s")
    for i, (doc, score) in enumerate(compressed_docs, 1):
        chars = len(doc.page_content)
        print(f"    [{i}] {chars} chars")

    # Step 5: Validate
    print("\n[Step 5] Validation...")
    t0 = time.time()
    final_docs = validate_compression(query, trimmed_docs, compressed_docs)
    t1 = time.time()
    print(f"  Validated in {t1-t0:.2f}s")

    # Format output
    result = []
    for i, (doc, score) in enumerate(final_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        result.append(f"[Document {i} - {source}]\n{doc.page_content}")

    final_text = "\n\n---\n\n".join(result)

    # Metrics
    original_chars = sum(len(d.page_content) for d, _ in scored_docs)
    final_chars = sum(len(d.page_content) for d, _ in final_docs)
    reduction = 100 - int(100 * final_chars / original_chars) if original_chars > 0 else 0

    return {
        "query": query,
        "chunks_retrieved": len(docs),
        "chunks_after_rerank": len(scored_docs),
        "chunks_after_filter": len(scored_docs),
        "chunks_final": len(final_docs),
        "original_chars": original_chars,
        "final_chars": final_chars,
        "reduction_percent": reduction,
        "final_text": final_text,
    }


def test_pipeline():
    print("Testing full RAG pipeline...")

    # Test with a query that should find relevant content
    query = "What is LangChain and how does it works"

    result = run_pipeline(query)

    if "error" in result:
        print(f"\n[FAIL] Pipeline error: {result['error']}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("Pipeline Results:")
    print(f"{'='*60}")
    print(f"  Chunks retrieved:    {result['chunks_retrieved']}")
    print(f"  Chunks after rerank: {result['chunks_after_rerank']}")
    print(f"  Chunks after filter: {result['chunks_after_filter']}")
    print(f"  Final chunks:        {result['chunks_final']}")
    print(f"  Original chars:      {result['original_chars']}")
    print(f"  Final chars:         {result['final_chars']}")
    print(f"  Reduction:           {result['reduction_percent']}%")
    print(f"\nFinal context preview (first 500 chars):")
    print(f"  {result['final_text'][:500]}...")

    # Assertions
    assert result['chunks_retrieved'] > 0, "Should retrieve at least one chunk"
    assert result['chunks_final'] > 0, "Should have at least one final chunk"
    assert result['final_chars'] > 0, "Final context should not be empty"
    assert result['reduction_percent'] >= 0, "Reduction should be non-negative"

    print("\n[OK] Full pipeline test passed!")


if __name__ == "__main__":
    try:
        test_pipeline()
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
