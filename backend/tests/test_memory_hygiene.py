"""Memory writer hygiene: upsert un-deletes, listing stops stamping, and
derived-state keys are blocked at save time.

Writes to the real Postgres database under a throwaway user_id and hard-deletes
only those rows, so no real user's memories are touched.
"""
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND.parent / ".env")

from memory import (
    delete_memory,
    get_memory_pool,
    get_user_memories,
    save_memory,
    touch_memories,
)
import agent_graph

TEST_USER = "hygiene-test-user"
FAILS = []


def check(label, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}{(' -> ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(label)


def rows_for(key):
    with get_memory_pool().connection() as conn:
        return conn.execute(
            "SELECT memory_id, value, deleted FROM memories "
            "WHERE user_id = %s AND key = %s",
            (TEST_USER, key),
        ).fetchall()


def stamp(memory_id):
    return {m["memory_id"]: m["last_accessed"] for m in get_user_memories(TEST_USER)}.get(memory_id)


print("\n=== fix 2: saving over a soft-deleted memory brings it back ===")
try:
    id1 = save_memory(TEST_USER, "preference", "primary_language", "Python", 0.7)
    check("first save returns an id", bool(id1))

    deleted = delete_memory(TEST_USER, id1)
    check("soft delete reports success", deleted)
    check("deleted memory is invisible", get_user_memories(TEST_USER) == [])

    id2 = save_memory(TEST_USER, "preference", "primary_language", "Go", 0.8)
    live = get_user_memories(TEST_USER)
    check("re-saving the same key makes it visible again", len(live) == 1, f"got {len(live)}")
    check("upsert reused the existing row", id2 == id1)
    if live:
        check("value is the new one", live[0]["value"] == "Go", live[0]["value"])
    check("no second row was created", len(rows_for("primary_language")) == 1)

    print("\n=== fix 3: listing no longer stamps every row ===")
    other = save_memory(TEST_USER, "context", "role", "Backend developer", 0.6)
    before = {m["memory_id"]: m["last_accessed"] for m in get_user_memories(TEST_USER)}
    get_user_memories(TEST_USER)
    get_user_memories(TEST_USER)
    after = {m["memory_id"]: m["last_accessed"] for m in get_user_memories(TEST_USER)}
    check("listing leaves last_accessed untouched", before == after)

    touched = touch_memories(TEST_USER, [id1])
    check("touch_memories reports one row", touched == 1, str(touched))
    now = {m["memory_id"]: m["last_accessed"] for m in get_user_memories(TEST_USER)}
    check("touched row advanced", now[id1] != before[id1])
    check("untouched row did not advance", now[other] == before[other])
    check("touch with no ids is a no-op", touch_memories(TEST_USER, []) == 0)

    print("\n=== fix 4: derived-state keys are blocked at save time ===")
    saved = agent_graph.save_extracted_memories(TEST_USER, [
        {"category": "fact", "key": "uploaded_files", "value": "a.pdf", "importance": 0.9},
        {"category": "fact", "key": "attached_document_scope", "value": "b.pdf", "importance": 0.9},
        {"category": "personal", "key": "name", "value": "Testperson", "importance": 0.7},
    ])
    check("blocked keys are skipped, allowed one saved", saved == 1, f"returned {saved}")
    check("uploaded_files never reached the store", rows_for("uploaded_files") == [])
    check("attached_document_scope never reached the store", rows_for("attached_document_scope") == [])
    check("allowed key saved normally", rows_for("name") != [])

    print("\n=== integration: the prompt helper stamps what it injects ===")
    before_lang = stamp(id1)
    block = agent_graph.load_memories_for_prompt(TEST_USER, "which programming language do I use")
    check("helper returns a memory block", "Go" in block, repr(block[:60]))
    check("helper stamped the memory it used", stamp(id1) != before_lang)
    check("empty user id yields empty block", agent_graph.load_memories_for_prompt("", "x") == "")
finally:
    with get_memory_pool().connection() as conn:
        conn.execute("DELETE FROM memories WHERE user_id = %s", (TEST_USER,))
        conn.execute("DELETE FROM users WHERE user_id = %s", (TEST_USER,))
    remaining = get_user_memories(TEST_USER)
    print(f"\ncleanup: test rows removed ({len(remaining)} left for {TEST_USER})")

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"FAILURES: {FAILS}"))
sys.exit(1 if FAILS else 0)
