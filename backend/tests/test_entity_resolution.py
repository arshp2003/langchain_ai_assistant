"""Entity resolution and the low-confidence fallback in search_documents.

Read-only against the real Chroma collection and the real memory store. The one
LLM call in this path (multi-query expansion) is stubbed, so the suite spends
zero requests - it asserts that too.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND.parent / ".env")

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END

import query_transform
import agent_graph
from agent_graph import AgentState, resolve_self_reference, search_documents, tool_node
from memory import get_user_memories

query_transform.expand_queries = lambda q, n=3: [q]

USER = "user-123"
RESUME = "Arsh_GenAI.pdf"
FAILS = []

graph = StateGraph(AgentState)
graph.add_node("tools", tool_node)
graph.add_edge(START, "tools")
graph.add_edge("tools", END)
app = graph.compile()


def check(label, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}{(' -> ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(label)


def ask(query, user_id=USER, filenames=(RESUME,)):
    msg = AIMessage(content="", tool_calls=[{
        "name": "search_documents",
        "args": {"query": query, "filenames": list(filenames)},
        "id": "c1",
    }])
    return app.invoke({
        "messages": [msg], "user_id": user_id, "attempts": 0,
        "validation_error": "", "user_memories": "",
        "needs_approval": False, "pending_tool_calls": [],
    })["messages"][-1]


names = [
    m["value"].strip()
    for m in get_user_memories(USER)
    if m["category"] == "personal" and m["key"] == "name"
]
NAME = names[0] if names else None

print("\n=== the injected state must stay invisible to the model ===")
check("tool schema exposes only query and filenames",
      sorted(search_documents.args) == ["filenames", "query"], str(sorted(search_documents.args)))

print("\n=== resolver ===")
check("no self-reference leaves the query alone",
      resolve_self_reference(USER, "programming language") is None)
check("no user id leaves the query alone",
      resolve_self_reference("", "my programming language") is None)
check("unknown user leaves the query alone",
      resolve_self_reference("no-such-user", "my programming language") is None)
if NAME:
    resolved = resolve_self_reference(USER, "my programming language")
    check("possessive is replaced by the stored name",
          resolved == f"{NAME} programming language", repr(resolved))
    check("the word 'my' is gone", " my " not in f" {resolved.lower()} ")
else:
    print(f"  [SKIP] no personal.name memory for {USER}; resolver has nothing to substitute")

print("\n=== the question that started this ===")
if NAME:
    r = ask("my programming language")
    check("returns document content, not a veto",
          "[Document" in r.content, repr(r.content[:70]))
    check("does not report nothing found",
          "No relevant documents found" not in r.content)
    check("the answer is actually present", "Python" in r.content)
else:
    print("  [SKIP] needs personal.name in the store")

print("\n=== the low-confidence fallback ===")
r2 = ask("my favourite cooking ingredient")
check("weak match is labelled, not discarded",
      r2.content.startswith("LOW CONFIDENCE"), repr(r2.content[:60]))
check("the label carries the score", "best score" in r2.content)
check("the closest chunk still reaches the model", "Arsh_GenAI.pdf" in r2.content)

print("\n=== cost ===")
requests_used = agent_graph.LLM_STATS["requests"]
check("zero OpenRouter requests spent", requests_used == 0, str(requests_used))

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
