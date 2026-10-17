import uuid

from langchain_core.messages import AIMessage, HumanMessage

from agent_graph import (
    MAX_ATTEMPTS,
    AgentState,
    builder,
    get_agent,
    get_thread_state,
    refine_node,
    resume_agent,
    route_after_refine,
    route_after_validation,
    stream_agent,
    text_of,
    thread_is_resumable,
    validate_answer_node,
)


# -----------------------------
# 1. Print graph topology
# -----------------------------

print("\n========== NODES ==========\n")
for name in builder.nodes:
    print(f"  {name}")

print("\n========== EDGES ==========\n")
for start, end in sorted(builder.edges):
    print(f"  {start} -> {end}")

print("\n========== CONDITIONAL EDGES ==========\n")
for start, branches in builder.branches.items():
    for branch in branches.values():
        path = getattr(branch.path, "func", branch.path)
        router = getattr(path, "__name__", str(path))
        targets = ", ".join(str(t) for t in branch.ends) if branch.ends else "chosen at runtime"
        print(f"  {start} --[{router}]--> {targets}")


# -----------------------------
# 2. Run graph and print trace (checkpointed, auto-recover on failure)
# -----------------------------

question = "What is (25 * 10) + 17, and what is that divided by 3?"

# A unique-per-execution thread id. Every node's state is saved to
# PostgreSQL under this id, so the run can be resumed if it stalls.
thread_id = f"demo-{uuid.uuid4().hex[:8]}"

print("\n========== EXECUTION TRACE ==========\n")
print(f"thread_id -> {thread_id}")

answer = ""


def print_event(item: dict) -> None:
    global answer

    print(f"\n--- node: {item['node']} ---")

    if item["event"] == "tool_calls":
        for call in item["tool_calls"]:
            print(f"tool call -> {call['name']}({call['args']})")

    elif item["event"] == "answer":
        print(item["content"])
        answer = item["content"]

    elif item["event"] == "validation":
        status = item["status"] if item["status"] == "VALID" else f"{item['status']}: {item['reason']}"
        print(f"[validate_answer] {status}")

    elif item["event"] == "retry":
        print(f"attempts -> {item['attempts']}")


try:
    for item in stream_agent(question, thread_id):
        print_event(item)
except Exception as exc:
    # Technical failure mid-run: pick up from the last saved checkpoint
    # instead of losing the progress made so far.
    print(f"\n[recovery] run failed ({exc!r}); resuming thread '{thread_id}' from checkpoint")
    for item in resume_agent(thread_id):
        print_event(item)


# -----------------------------
# 3. Print final response
# -----------------------------

print("\n========== FINAL ANSWER ==========\n")

print(answer)


# -----------------------------
# 4. Inspect the saved checkpoint
# -----------------------------

print("\n========== CHECKPOINT STATE ==========\n")

config = {"configurable": {"thread_id": thread_id}}
snapshot = get_thread_state(thread_id)
history = list(get_agent().get_state_history(config))

print("checkpoint backend -> PostgreSQL")
print(f"saved checkpoints  -> {len(history)}")
print(f"pending next nodes -> {snapshot.next or '(finished)'}")
print(f"resumable          -> {thread_is_resumable(thread_id)}")
print(f"persisted messages -> {len(snapshot.values.get('messages', []))}")
print(f"persisted attempts -> {snapshot.values.get('attempts')}")


# -----------------------------
# 5. Resume demo (recover a stuck thread)
# -----------------------------

print("\n========== RESUME DEMO ==========\n")

if thread_is_resumable(thread_id):
    print(f"thread '{thread_id}' has pending work -> resuming from checkpoint")
    for item in resume_agent(thread_id):
        print_event(item)
else:
    print(
        f"thread '{thread_id}' finished cleanly -> nothing to resume "
        "(resume_agent would be a safe no-op)"
    )


# -----------------------------
# 6. Demo the invalid path
# -----------------------------

print("\n========== INVALID PATH DEMO ==========\n")

bad_state: AgentState = {
    "messages": [
        HumanMessage(content=question),
        AIMessage(content="I'm unable to help with that."),
    ],
    "attempts": 0,
    "validation_error": "",
}

print(f"route_after_validation -> {route_after_validation(validate_answer_node(bad_state))}")

refined_state = refine_node(bad_state)

print(f"route_after_refine -> {route_after_refine(refined_state)}")
print(f"attempts -> {refined_state['attempts']}")
print(f"answer -> {text_of(refined_state['messages'][-1])}")

print(f"\nroute_after_refine -> {route_after_refine({**bad_state, 'attempts': MAX_ATTEMPTS})}")