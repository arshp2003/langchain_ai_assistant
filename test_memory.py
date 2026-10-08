"""Test script to demonstrate long-term memory system."""
import os
import sys
from dotenv import load_dotenv

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

from agent_graph import run_agent
from memory import get_user_memories, init_memory_tables


def test_memory_system():
    """Test that memories are extracted and loaded correctly."""
    print("=" * 60)
    print("Testing Long-Term Memory System")
    print("=" * 60)

    # Initialize memory tables
    print("\n1. Initializing memory tables...")
    init_memory_tables()
    print("   [OK] Tables created")

    # Test user ID
    user_id = "test-user-001"

    # First conversation - should extract memories
    print(f"\n2. First conversation with user '{user_id}'...")
    print("   Question: 'I'm a data scientist named Alice and I prefer Python for ML work'")

    result1 = run_agent(
        "I'm a data scientist named Alice and I prefer Python for ML work",
        user_id=user_id,
    )

    print(f"   Answer: {result1['answer'][:100]}...")
    print(f"   LLM requests: {result1['llm']['requests']}")

    # Check extracted memories
    print(f"\n3. Checking extracted memories for user '{user_id}'...")
    memories = get_user_memories(user_id)
    print(f"   Found {len(memories)} memories:")
    for mem in memories:
        print(f"   - [{mem['category']}] {mem['key']}: {mem['value']} (importance: {mem['importance']})")

    # Second conversation - should use memories
    print(f"\n4. Second conversation with same user...")
    print("   Question: 'What programming language should I use for my next project?'")

    result2 = run_agent(
        "What programming language should I use for my next project?",
        user_id=user_id,
    )

    print(f"   Answer: {result2['answer'][:100]}...")
    print(f"   LLM requests: {result2['llm']['requests']}")
    print("   (The agent should remember the user prefers Python)")

    # Summary
    print("\n" + "=" * 60)
    print("Memory System Test Complete")
    print("=" * 60)
    print(f"Total memories stored: {len(memories)}")
    print(f"Total LLM requests: {result1['llm']['requests'] + result2['llm']['requests']}")
    print("\nNote: Memory extraction adds 1 extra LLM call per conversation.")
    print("This is the cost of automatic memory extraction.")


if __name__ == "__main__":
    test_memory_system()
