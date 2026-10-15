import os
import random
import threading
import time
import uuid
from typing import Annotated, TypedDict

from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from openrouter.errors import (
    BadGatewayResponseError,
    EdgeNetworkTimeoutResponseError,
    InternalServerResponseError,
    NoResponseError,
    ProviderOverloadedResponseError,
    RequestTimeoutResponseError,
    ResponseValidationError,
    ServiceUnavailableResponseError,
    TooManyRequestsResponseError,
)
from psycopg_pool import ConnectionPool

from memory import (
    format_memories_for_prompt,
    init_memory_tables,
    register_thread,
    save_memory,
    semantic_search,
)


# Configurable via .env (OPENROUTER_MODEL) so you can swap slugs without editing code.
MODEL_NAME = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-ultra-550b-a55b:free")

MAX_ATTEMPTS = 2


# -----------------------------
# 1. Define tools
# -----------------------------

@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def divide_numbers(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


tools = [
    add_numbers,
    multiply_numbers,
    divide_numbers,
]


# -----------------------------
# 2. Create models
# -----------------------------

model = ChatOpenRouter(
    model=MODEL_NAME
)

model_with_tools = model.bind_tools(tools)


# -----------------------------
# 3. Define state
# -----------------------------

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    attempts: int
    validation_error: str
    user_id: str
    user_memories: str


# -----------------------------
# 4. Helper functions
# -----------------------------

REFUSAL_PHRASES = (
    "i cannot",
    "i can't",
    "i don't know",
    "i do not know",
    "as an ai",
    "i'm unable",
)


def text_of(message: BaseMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    return " ".join(
        block.get("text", "")
        for block in message.content
        if isinstance(block, dict)
    )


def check_answer(answer: str) -> str:
    if not answer.strip():
        return "the answer was empty"
    if any(phrase in answer.lower() for phrase in REFUSAL_PHRASES):
        return "the answer was a refusal, not a real answer"
    if len(answer.strip()) < 3:
        return "the answer was too short to be useful"
    return ""


# Transient OpenRouter errors worth retrying: free-tier rate limits (429),
# overloaded providers, timeouts and 5xx. Auth / model-not-found / payment
# errors are NOT retried - a second try would fail the same way and just waste
# requests against the free daily cap.
#
# ResponseValidationError is included because when a free provider is overloaded
# OpenRouter sometimes returns an error-shaped body that the SDK cannot parse
# into a typed ProviderOverloadedResponseError, surfacing instead as a generic
# validation error. It is the same transient overload, so it should retry too.
RETRYABLE_ERRORS = (
    TooManyRequestsResponseError,
    ProviderOverloadedResponseError,
    InternalServerResponseError,
    ServiceUnavailableResponseError,
    BadGatewayResponseError,
    RequestTimeoutResponseError,
    EdgeNetworkTimeoutResponseError,
    NoResponseError,
    ResponseValidationError,
)

MAX_RETRIES = 4         # total attempts per model call
RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt (2, 4, 8, ...)
RETRY_MAX_DELAY = 20.0  # cap so a single wait never gets too long

# Every model request funnels through invoke_model, so that is the only place
# retries are visible. A 429 or timeout never produces an AIMessage, so counting
# trace events cannot tell a node that cost 1 request from one that cost 4 - and
# on a free tier those wasted attempts are exactly what burns the daily cap.
LLM_STATS = {
    "requests": 0,
    "failures": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "per_model": {},
}

_stats_lock = threading.Lock()


def _model_slug(runnable) -> str:
    """Model id for a chat model, unwrapping a bind_tools() binding if needed."""
    target = getattr(runnable, "bound", runnable)
    return getattr(target, "model_name", "unknown")


def _record(requests: int = 0, failures: int = 0, usage: dict | None = None,
            slug: str = "") -> None:
    with _stats_lock:
        LLM_STATS["requests"] += requests
        LLM_STATS["failures"] += failures
        if usage:
            LLM_STATS["prompt_tokens"] += usage.get("input_tokens", 0)
            LLM_STATS["completion_tokens"] += usage.get("output_tokens", 0)
        if requests:
            LLM_STATS["per_model"][slug] = LLM_STATS["per_model"].get(slug, 0) + requests


def get_llm_stats() -> dict:
    """Cumulative model request counts since process start or last reset."""
    with _stats_lock:
        return {**LLM_STATS, "per_model": dict(LLM_STATS["per_model"])}


def reset_llm_stats() -> None:
    """Zero the cumulative counters (does not touch any single run's delta)."""
    with _stats_lock:
        LLM_STATS.update(
            requests=0,
            failures=0,
            prompt_tokens=0,
            completion_tokens=0,
            per_model={},
        )


def _stats_delta(before: dict) -> dict:
    """Requests attributable to one run, given a snapshot taken before it."""
    return {
        "requests": LLM_STATS["requests"] - before["requests"],
        "failures": LLM_STATS["failures"] - before["failures"],
        "prompt_tokens": LLM_STATS["prompt_tokens"] - before["prompt_tokens"],
        "completion_tokens": LLM_STATS["completion_tokens"] - before["completion_tokens"],
    }


def extract_memories_from_conversation(messages: list[BaseMessage]) -> list[dict]:
    """Use LLM to extract personal info and preferences from conversation."""
    conversation_text = "\n".join(
        f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {text_of(msg)}"
        for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
    )

    extraction_prompt = f"""Analyze this conversation and extract any personal information, preferences, or facts about the user that should be remembered for future conversations.

Conversation:
{conversation_text}

Extract memories in these categories:
- personal: name, age, location, family, etc.
- preference: likes, dislikes, preferred tools/languages, etc.
- context: job role, projects, goals, etc.
- fact: any other relevant facts about the user

Return a JSON array of objects with this structure:
[
  {{"category": "personal|preference|context|fact", "key": "short_key", "value": "the information", "importance": 0.0-1.0}}
]

If no memories should be extracted, return an empty array: []

Only extract information that is:
1. About the user (not general knowledge)
2. Likely to be useful in future conversations
3. Not already obvious from the current conversation

Return ONLY the JSON array, no other text."""

    response = invoke_model(model, [HumanMessage(content=extraction_prompt)])

    try:
        import json
        response_text = text_of(response).strip()
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()

        memories = json.loads(response_text)
        if isinstance(memories, list):
            return [
                {
                    "category": m.get("category", "fact"),
                    "key": m.get("key", ""),
                    "value": m.get("value", ""),
                    "importance": float(m.get("importance", 0.5)),
                }
                for m in memories
                if m.get("key") and m.get("value")
            ]
    except Exception as exc:
        print(f"[memory] failed to extract memories: {exc}")

    return []


def save_extracted_memories(user_id: str, memories: list[dict]) -> int:
    """Save extracted memories to database, returns count of memories saved."""
    if not user_id or not memories:
        return 0

    count = 0
    for mem in memories:
        try:
            save_memory(
                user_id=user_id,
                category=mem["category"],
                key=mem["key"],
                value=mem["value"],
                importance=mem["importance"],
            )
            count += 1
        except Exception as exc:
            print(f"[memory] failed to save memory {mem}: {exc}")

    return count


def invoke_model(runnable, messages):
    """Call a model, retrying transient OpenRouter errors with backoff + jitter."""
    slug = _model_slug(runnable)
    delay = RETRY_BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        _record(requests=1, slug=slug)
        try:
            response = runnable.invoke(messages)
            _record(usage=getattr(response, "usage_metadata", None))
            return response
        except RETRYABLE_ERRORS as exc:
            _record(failures=1)
            if attempt == MAX_RETRIES:
                raise
            wait = min(delay, RETRY_MAX_DELAY) + random.uniform(0, 1)
            print(
                f"[retry] {type(exc).__name__} on attempt {attempt}/{MAX_RETRIES}; "
                f"waiting {wait:.1f}s before retry"
            )
            time.sleep(wait)
            delay *= 2


# -----------------------------
# 5. Define nodes
# -----------------------------

def agent_node(state: AgentState) -> dict:
    response = invoke_model(model_with_tools, state["messages"])
    return {"messages": [response]}


def validate_answer_node(state: AgentState) -> dict:
    return {"validation_error": check_answer(text_of(state["messages"][-1]))}


def refine_node(state: AgentState) -> dict:
    prompt = (
        "Your previous answer was rejected because "
        f"{state.get('validation_error', 'it was not usable')}. "
        "Give a corrected, self-contained final answer. "
        "Do not ask for more information."
    )
    response = invoke_model(
        model, [*state["messages"], HumanMessage(content=prompt)]
    )
    return {
        "messages": [response],
        "attempts": state.get("attempts", 0) + 1,
        "validation_error": "",
    }


def fallback_node(state: AgentState) -> dict:
    prompt = (
        "Answer the question directly in one short sentence. "
        "Do not ask questions and do not refuse."
    )
    response = invoke_model(
        model, [*state["messages"], HumanMessage(content=prompt)]
    )
    return {"messages": [response], "validation_error": ""}


def extract_memory_node(state: AgentState) -> dict:
    """Extract and save user memories from the conversation."""
    user_id = state.get("user_id")
    if not user_id:
        return {}

    try:
        memories = extract_memories_from_conversation(state["messages"])
        if memories:
            count = save_extracted_memories(user_id, memories)
            print(f"[memory] extracted and saved {count} memories for user {user_id}")
    except Exception as exc:
        print(f"[memory] extraction failed: {exc}")

    return {}


tool_node = ToolNode(tools)


# -----------------------------
# 6. Define edges (routers)
# -----------------------------

def route_after_agent(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return "validate_answer"


def route_after_validation(state: AgentState) -> str:
    if state.get("validation_error"):
        return "refine"
    return "extract_memory"


def route_after_refine(state: AgentState) -> str:
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "fallback"
    return "validate_answer"


# -----------------------------
# 7. Build graph (with PostgreSQL checkpointing)
# -----------------------------

builder = StateGraph(AgentState)

builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("validate_answer", validate_answer_node)
builder.add_node("refine", refine_node)
builder.add_node("fallback", fallback_node)
builder.add_node("extract_memory", extract_memory_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_after_agent)
builder.add_edge("tools", "agent")
builder.add_conditional_edges("validate_answer", route_after_validation)
builder.add_conditional_edges("refine", route_after_refine)
builder.add_edge("fallback", "extract_memory")
builder.add_edge("extract_memory", END)


# Checkpoints are persisted to PostgreSQL so a run can be recovered even after
# the process crashes or is restarted. Set POSTGRES_URI in your .env, e.g.
#   POSTGRES_URI=postgresql://user:password@localhost:5432/dbname?sslmode=disable
#
# The checkpointer and compiled graph are built lazily on first use, so
# importing this module (or booting the API) does not require a live database.
_agent = None
_checkpointer = None


def get_checkpointer() -> PostgresSaver:
    global _checkpointer
    if _checkpointer is None:
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise RuntimeError(
                "POSTGRES_URI is not set. Add it to your .env file, e.g. "
                "POSTGRES_URI=postgresql://user:password@localhost:5432/dbname"
            )
        pool = ConnectionPool(
            conninfo=uri,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
        saver = PostgresSaver(pool)
        saver.setup()
        init_memory_tables()
        _checkpointer = saver
    return _checkpointer


def get_agent():
    global _agent
    if _agent is None:
        _agent = builder.compile(checkpointer=get_checkpointer())
    return _agent


_agent_hitl = None


def get_agent_hitl():
    """Compiled graph that pauses before tool execution and memory extraction.

    Uses interrupt_before so the graph saves its checkpoint and waits for
    human approval before running those nodes. The state at the interrupt
    point shows exactly what the agent plans to do next.
    """
    global _agent_hitl
    if _agent_hitl is None:
        _agent_hitl = builder.compile(
            checkpointer=get_checkpointer(),
            interrupt_before=["tools", "extract_memory"],
        )
    return _agent_hitl


def get_pending_approval(thread_id: str) -> dict:
    """Inspect what the graph wants to do next while paused at an interrupt.

    Returns the pending node name and a human-readable description of the
    action awaiting approval, so the user can decide to approve or reject.
    """
    config = _thread_config(thread_id)
    snapshot = get_agent_hitl().get_state(config)

    if not snapshot.next:
        return {"status": "no_pending_action", "thread_id": thread_id}

    pending_node = snapshot.next[0]
    values = snapshot.values
    last_messages = values.get("messages", [])
    last_message = last_messages[-1] if last_messages else None

    detail = {}
    if pending_node == "tools" and last_message and isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", [])
        detail = {
            "pending_node": "tools",
            "action": "execute_tools",
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]}
                for tc in tool_calls
            ],
        }
    elif pending_node == "extract_memory":
        detail = {
            "pending_node": "extract_memory",
            "action": "extract_and_save_memories",
            "message": "Agent will analyze the conversation and save personal info/preferences to long-term memory.",
        }
    else:
        detail = {
            "pending_node": pending_node,
            "action": pending_node,
        }

    detail["thread_id"] = thread_id
    detail["status"] = "awaiting_approval"
    return detail


def resume_with_approval(thread_id: str, approved: bool, feedback: str = "") -> dict:
    """Resume a paused graph after human review.

    If approved, the pending node runs normally.
    If rejected, a human feedback message is injected so the agent can
    adjust its behaviour, and the graph continues from there.
    """
    config = _thread_config(thread_id)
    graph = get_agent_hitl()
    snapshot = graph.get_state(config)

    if not snapshot.next:
        return {"status": "nothing_to_resume", "thread_id": thread_id}

    pending_node = snapshot.next[0]

    if not approved:
        rejection_msg = f"Human reviewer rejected the '{pending_node}' step."
        if feedback:
            rejection_msg += f" Feedback: {feedback}"

        if pending_node == "tools":
            values = snapshot.values
            last_message = values["messages"][-1]
            tool_call_ids = [
                tc["id"] for tc in getattr(last_message, "tool_calls", [])
            ]
            from langchain_core.messages import ToolMessage
            rejection_messages = [
                ToolMessage(
                    content=rejection_msg,
                    tool_call_id=tc_id,
                )
                for tc_id in tool_call_ids
            ]
            graph.update_state(config, {"messages": rejection_messages})
        else:
            graph.update_state(
                config,
                {"messages": [HumanMessage(content=rejection_msg)]},
            )

    trace = []
    answer = ""
    for item in _events_from(graph.stream(None, config, stream_mode="updates")):
        trace.append(item)
        if item["event"] == "answer":
            answer = item["content"]

    return {
        "status": "resumed",
        "approved": approved,
        "thread_id": thread_id,
        "answer": answer,
        "trace": trace,
        "still_pending": bool(graph.get_state(config).next),
    }


# -----------------------------
# 8. Run graph as event stream (checkpointed + resumable)
# -----------------------------

def _thread_config(thread_id: str | None) -> dict:
    """Build a LangGraph config routing a run to a checkpoint thread.

    A missing thread_id gets a fresh unique id so independent runs never
    share (and append to) the same checkpointed message history.
    """
    return {"configurable": {"thread_id": thread_id or f"run-{uuid.uuid4().hex}"}}


def _events_from(stream):
    """Translate raw graph update steps into the trace events callers expect."""
    for step in stream:

        for node, update in step.items():

            if update is None:
                continue

            if not isinstance(update, dict):
                continue

            for message in update.get("messages", []):

                if isinstance(message, AIMessage) and message.tool_calls:
                    yield {
                        "node": node,
                        "event": "tool_calls",
                        "tool_calls": [
                            {"name": call["name"], "args": call["args"]}
                            for call in message.tool_calls
                        ],
                    }
                else:
                    yield {
                        "node": node,
                        "event": "answer",
                        "content": text_of(message),
                    }

            if node == "validate_answer":
                problem = update.get("validation_error", "")
                yield {
                    "node": node,
                    "event": "validation",
                    "status": "VALID" if not problem else "INVALID",
                    "reason": problem,
                }

            if "attempts" in update:
                yield {
                    "node": node,
                    "event": "retry",
                    "attempts": update["attempts"],
                }


def stream_agent(question: str, thread_id: str | None = None, user_id: str | None = None):
    """Run a fresh question, saving every step to the thread's checkpoint."""
    user_memories = ""
    if user_id:
        try:
            memories = semantic_search(user_id, question, top_k=3)
            user_memories = format_memories_for_prompt(memories)
        except Exception as exc:
            print(f"[memory] failed to search memories: {exc}")

    messages = [HumanMessage(content=question)]
    if user_memories:
        system_msg = f"System: {user_memories}\n\n"
        messages.insert(0, HumanMessage(content=system_msg))

    inputs: AgentState = {
        "messages": messages,
        "attempts": 0,
        "validation_error": "",
        "user_id": user_id or "",
        "user_memories": user_memories,
    }
    config = _thread_config(thread_id)
    yield from _events_from(get_agent().stream(inputs, config, stream_mode="updates"))


def resume_agent(thread_id: str):
    """Continue a stuck/failed thread from its last saved checkpoint.

    Passing None as the input tells LangGraph to pick up where the thread
    stopped (re-running the pending node) instead of starting a new run.
    """
    config = _thread_config(thread_id)
    yield from _events_from(get_agent().stream(None, config, stream_mode="updates"))


def get_thread_state(thread_id: str):
    """Return the saved StateSnapshot for a thread (values, next nodes, ...)."""
    return get_agent().get_state(_thread_config(thread_id))


def thread_is_resumable(thread_id: str) -> bool:
    """True if the thread has a saved checkpoint with work still pending."""
    snapshot = get_thread_state(thread_id)
    return bool(snapshot.values) and bool(snapshot.next)


def run_agent(question: str, thread_id: str | None = None, user_id: str | None = None) -> dict:
    trace = []
    answer = ""
    before = get_llm_stats()

    config = _thread_config(thread_id)
    resolved_thread_id = config["configurable"]["thread_id"]

    user_memories = ""
    if user_id:
        try:
            memories = semantic_search(user_id, question, top_k=3)
            user_memories = format_memories_for_prompt(memories)
        except Exception as exc:
            print(f"[memory] failed to search memories: {exc}")

    messages = [HumanMessage(content=question)]
    if user_memories:
        system_msg = f"System: {user_memories}\n\n"
        messages.insert(0, HumanMessage(content=system_msg))

    inputs: AgentState = {
        "messages": messages,
        "attempts": 0,
        "validation_error": "",
        "user_id": user_id or "",
        "user_memories": user_memories,
    }

    for item in _events_from(get_agent().stream(inputs, config, stream_mode="updates")):
        trace.append(item)
        if item["event"] == "answer":
            answer = item["content"]

    if user_id:
        try:
            title = question[:50] if len(question) > 50 else question
            register_thread(user_id, resolved_thread_id, title)
        except Exception as exc:
            print(f"[memory] failed to register thread: {exc}")

    return {
        "answer": answer,
        "trace": trace,
        "thread_id": resolved_thread_id,
        "llm": _stats_delta(before),
    }


def run_agent_with_recovery(
    question: str,
    thread_id: str | None = None,
    user_id: str | None = None,
    max_resumes: int = 3,
) -> dict:
    """Run the agent and, if a node fails mid-run, resume from the checkpoint.

    Every completed node is persisted, so a transient technical failure (API
    timeout, crash, etc.) does not lose progress: the pending node is retried
    from the last good checkpoint instead of restarting the whole run.
    """
    config = _thread_config(thread_id)
    resolved_thread_id = config["configurable"]["thread_id"]

    trace: list = []
    answer = ""
    resumed = False
    before = get_llm_stats()

    def consume(events) -> None:
        nonlocal answer
        for item in events:
            trace.append(item)
            if item["event"] == "answer":
                answer = item["content"]

    user_memories = ""
    if user_id:
        try:
            memories = semantic_search(user_id, question, top_k=3)
            user_memories = format_memories_for_prompt(memories)
        except Exception as exc:
            print(f"[memory] failed to search memories: {exc}")

    messages = [HumanMessage(content=question)]
    if user_memories:
        system_msg = f"System: {user_memories}\n\n"
        messages.insert(0, HumanMessage(content=system_msg))

    inputs: AgentState = {
        "messages": messages,
        "attempts": 0,
        "validation_error": "",
        "user_id": user_id or "",
        "user_memories": user_memories,
    }

    try:
        consume(_events_from(get_agent().stream(inputs, config, stream_mode="updates")))
    except Exception as exc:
        print(f"[recovery] run failed ({exc!r}); resuming from checkpoint")
        for attempt in range(1, max_resumes + 1):
            snapshot = get_agent().get_state(config)
            if not snapshot.next:
                break
            resumed = True
            print(f"[recovery] resume {attempt}: pending nodes -> {snapshot.next}")
            try:
                consume(_events_from(get_agent().stream(None, config, stream_mode="updates")))
                break
            except Exception as resume_exc:
                print(f"[recovery] resume {attempt} failed ({resume_exc!r})")
        else:
            print("[recovery] exhausted resume attempts")

    return {
        "answer": answer,
        "trace": trace,
        "thread_id": resolved_thread_id,
        "resumed": resumed,
        "llm": _stats_delta(before),
    }


def run_agent_hitl(
    question: str,
    thread_id: str | None = None,
    user_id: str | None = None,
) -> dict:
    """Start a question using the human-in-the-loop graph.

    Runs until the first interrupt point (tool execution or memory extraction),
    then pauses and returns the pending action for human review. The caller
    uses get_pending_approval() to inspect and resume_with_approval() to
    approve or reject.
    """
    before = get_llm_stats()
    config = _thread_config(thread_id)
    resolved_thread_id = config["configurable"]["thread_id"]

    user_memories = ""
    if user_id:
        try:
            memories = semantic_search(user_id, question, top_k=3)
            user_memories = format_memories_for_prompt(memories)
        except Exception as exc:
            print(f"[memory] failed to search memories: {exc}")

    messages = [HumanMessage(content=question)]
    if user_memories:
        system_msg = f"System: {user_memories}\n\n"
        messages.insert(0, HumanMessage(content=system_msg))

    inputs: AgentState = {
        "messages": messages,
        "attempts": 0,
        "validation_error": "",
        "user_id": user_id or "",
        "user_memories": user_memories,
    }

    trace = []
    answer = ""
    graph = get_agent_hitl()

    for item in _events_from(graph.stream(inputs, config, stream_mode="updates")):
        trace.append(item)
        if item["event"] == "answer":
            answer = item["content"]

    snapshot = graph.get_state(config)
    pending = get_pending_approval(resolved_thread_id)

    if user_id:
        try:
            title = question[:50] if len(question) > 50 else question
            register_thread(user_id, resolved_thread_id, title)
        except Exception as exc:
            print(f"[memory] failed to register thread: {exc}")

    return {
        "answer": answer,
        "trace": trace,
        "thread_id": resolved_thread_id,
        "pending_approval": pending,
        "llm": _stats_delta(before),
    }