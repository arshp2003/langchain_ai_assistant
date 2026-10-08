import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agent_graph import (
    MAX_ATTEMPTS,
    MODEL_NAME,
    builder,
    get_llm_stats,
    reset_llm_stats,
    run_agent,
    stream_agent,
)
from memory import (
    delete_all_user_memories,
    delete_memory,
    get_user_memories,
    get_user_profile,
    save_memory,
)


app = FastAPI(
    title="Multi Node LangGraph Agent API",
    description="A ReAct agent with answer validation, retry and fallback nodes.",
    version="1.0.0",
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
    result = run_agent(request.question, user_id=request.user_id)
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)