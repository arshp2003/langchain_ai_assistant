import json
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_graph import (
    MAX_ATTEMPTS,
    MODEL_NAME,
    builder,
    get_llm_stats,
    get_pending_approval,
    reset_llm_stats,
    resume_with_approval,
    run_agent,
    run_agent_hitl,
    stream_agent,
)
from memory import (
    delete_all_user_memories,
    delete_memory,
    delete_thread,
    get_user_memories,
    get_user_profile,
    get_user_threads,
    save_memory,
)


app = FastAPI(
    title="Multi Node LangGraph Agent API",
    description="A ReAct agent with answer validation, retry and fallback nodes.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        examples=["What is (25 * 10) + 17, and what is that divided by 3?"],
    )
    user_id: str | None = Field(
        default=None,
        description="Optional user ID for long-term memory",
        examples=["user-123"],
    )
    attached_documents: list[str] | None = Field(
        default=None,
        description="Optional list of filenames to restrict document search to",
        examples=[["report.pdf", "notes.txt"]],
    )


class ChatResponse(BaseModel):
    question: str
    answer: str
    trace: list[dict]
    llm: dict = Field(default_factory=dict)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "max_attempts": MAX_ATTEMPTS,
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = run_agent(request.question, user_id=request.user_id, attached_documents=request.attached_documents)
    return ChatResponse(
        question=request.question,
        answer=result["answer"],
        trace=result["trace"],
        llm=result["llm"],
    )


@app.get("/usage")
def usage() -> dict:
    """Cumulative model requests sent since process start or last reset."""
    return get_llm_stats()


@app.post("/usage/reset")
def usage_reset() -> dict:
    reset_llm_stats()
    return get_llm_stats()


@app.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:

    def event_stream():
        for item in stream_agent(request.question, user_id=request.user_id):
            yield f"data: {json.dumps(item)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


class MemoryRequest(BaseModel):
    user_id: str = Field(..., examples=["user-123"])
    category: str = Field(..., examples=["preference"])
    key: str = Field(..., examples=["favorite_language"])
    value: str = Field(..., examples=["Python"])
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


@app.get("/memories/{user_id}")
def get_memories(user_id: str) -> dict:
    """Get all memories for a specific user."""
    memories = get_user_memories(user_id)
    return {
        "user_id": user_id,
        "count": len(memories),
        "memories": memories,
    }


@app.post("/memories")
def add_memory(request: MemoryRequest) -> dict:
    """Manually add a memory for a user."""
    memory_id = save_memory(
        user_id=request.user_id,
        category=request.category,
        key=request.key,
        value=request.value,
        importance=request.importance,
    )
    return {
        "status": "saved",
        "memory_id": memory_id,
        "user_id": request.user_id,
        "category": request.category,
        "key": request.key,
    }


@app.delete("/memories/{user_id}/{memory_id}")
def remove_memory(user_id: str, memory_id: str) -> dict:
    """Delete a specific memory."""
    deleted = delete_memory(user_id, memory_id)
    return {
        "status": "deleted" if deleted else "not_found",
        "user_id": user_id,
        "memory_id": memory_id,
    }


@app.delete("/memories/{user_id}")
def remove_all_memories(user_id: str) -> dict:
    """Delete all memories for a user."""
    count = delete_all_user_memories(user_id)
    return {
        "status": "deleted",
        "user_id": user_id,
        "deleted_count": count,
    }


@app.get("/user/{user_id}")
def user_profile(user_id: str) -> dict:
    """Get user profile information."""
    profile = get_user_profile(user_id)
    if not profile:
        return {"status": "not_found", "user_id": user_id}
    return profile


@app.get("/threads/{user_id}")
def list_threads(user_id: str, limit: int = 50) -> dict:
    """Get all conversation threads for a user."""
    threads = get_user_threads(user_id, limit)
    return {
        "user_id": user_id,
        "count": len(threads),
        "threads": threads,
    }


@app.delete("/threads/{user_id}/{thread_id}")
def remove_thread(user_id: str, thread_id: str) -> dict:
    """Delete a thread mapping for a user."""
    deleted = delete_thread(user_id, thread_id)
    return {
        "status": "deleted" if deleted else "not_found",
        "user_id": user_id,
        "thread_id": thread_id,
    }


@app.get("/graph")
def graph() -> dict:
    return {
        "nodes": list(builder.nodes),
        "edges": [list(edge) for edge in sorted(builder.edges)],
        "conditional_edges": [
            {
                "from": start,
                "router": getattr(
                    getattr(branch.path, "func", branch.path),
                    "__name__",
                    str(branch.path),
                ),
            }
            for start, branches in builder.branches.items()
            for branch in branches.values()
        ],
    }


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., examples=[True])
    feedback: str = Field(
        default="",
        description="Optional feedback when rejecting, e.g. 'use a different approach'",
    )


@app.post("/chat/hitl")
def chat_hitl(request: ChatRequest) -> dict:
    """Start a human-in-the-loop run. Pauses before tool calls and memory
    extraction so the user can review and approve/reject each step."""
    result = run_agent_hitl(request.question, user_id=request.user_id, attached_documents=request.attached_documents)
    return result


@app.get("/approval/{thread_id}")
def check_approval(thread_id: str) -> dict:
    """See what action the graph is waiting to execute next."""
    return get_pending_approval(thread_id)


@app.post("/approval/{thread_id}")
def handle_approval(thread_id: str, request: ApprovalRequest) -> dict:
    """Approve or reject the pending action and resume the graph."""
    return resume_with_approval(thread_id, request.approved, request.feedback)


# -----------------------------
# Document Management (RAG)
# -----------------------------

DOCUMENTS_DIR = Path(__file__).resolve().parent / "data" / "documents"
CHROMA_DIR = Path(__file__).resolve().parent / "chroma_db"


def _get_vector_store():
    from langchain_chroma import Chroma
    from embeddings import LocalEmbeddings

    embedding_function = LocalEmbeddings()
    return Chroma(
        collection_name="langchain_notes",
        embedding_function=embedding_function,
        persist_directory=str(CHROMA_DIR),
    )


def _load_and_chunk(file_path: Path):
    """Load a document file and split it into chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(file_path))
    elif suffix in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(file_path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")

    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)


@app.post("/documents/upload")
async def upload_document(file: UploadFile) -> dict:
    """Upload a document (PDF, TXT, or MD), chunk it, and store in the vector database."""
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    file_path = DOCUMENTS_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    try:
        chunks = _load_and_chunk(file_path)
        vector_store = _get_vector_store()

        for chunk in chunks:
            chunk.metadata["source"] = file.filename

        ids = vector_store.add_documents(chunks)

        return {
            "status": "uploaded",
            "filename": file.filename,
            "chunks_created": len(chunks),
            "vectors_stored": len(ids),
        }
    except Exception as exc:
        return {
            "status": "error",
            "filename": file.filename,
            "error": str(exc),
        }


@app.get("/documents")
def list_documents() -> dict:
    """List all uploaded documents."""
    if not DOCUMENTS_DIR.exists():
        return {"documents": [], "count": 0}

    documents = []
    for file_path in DOCUMENTS_DIR.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in (".pdf", ".txt", ".md"):
            stat = file_path.stat()
            documents.append({
                "filename": file_path.name,
                "size_bytes": stat.st_size,
                "type": file_path.suffix[1:].upper(),
            })

    return {
        "documents": sorted(documents, key=lambda d: d["filename"]),
        "count": len(documents),
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str) -> dict:
    """Delete a document file. Note: vector embeddings remain in ChromaDB."""
    file_path = DOCUMENTS_DIR / filename
    if not file_path.exists():
        return {"status": "not_found", "filename": filename}

    file_path.unlink()
    return {
        "status": "deleted",
        "filename": filename,
        "note": "File removed. Vector embeddings may still exist in ChromaDB.",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)