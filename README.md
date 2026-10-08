# LangChain AI Assistant with Long-Term Memory

A production-ready AI assistant built with LangGraph that combines short-term conversation persistence (checkpoints) with long-term user memory, featuring automatic memory extraction, retry logic, and recovery mechanisms.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Flow](#project-flow)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [Memory System](#memory-system)
- [Checkpoint System](#checkpoint-system)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)

## Overview

This project implements a multi-node ReAct agent with:

- **Short-term memory**: PostgreSQL checkpointing saves every conversation step, enabling recovery from failures
- **Long-term memory**: Automatic extraction and storage of user preferences, personal info, and context across conversations
- **Resilience**: Retry logic with exponential backoff handles transient API errors
- **Validation**: Multi-stage answer validation ensures quality responses
- **Observability**: Built-in tracking of LLM requests, tokens, and failures

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Request                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Load User Memories from Database                │
│         (if user_id provided, inject into context)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Agent Node                              │
│            (LLM decides: use tool or answer)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
    ┌─────────────────┐       ┌──────────────────┐
    │   Tool Node     │       │ Validate Answer  │
    │ (execute tool)  │       │      Node        │
    └────────┬────────┘       └────────┬─────────┘
             │                         │
             └─────────┐               │
                       │               ▼
                       │      ┌────────────────┐
                       │      │   Is Valid?    │
                       │      └───┬────────┬───┘
                       │          │        │
                       │         Yes       No
                       │          │        │
                       │          ▼        ▼
                       │   ┌──────────┐  ┌──────────┐
                       │   │ Extract  │  │  Refine  │
                       │   │ Memory   │  │   Node   │
                       │   │  Node    │  └────┬─────┘
                       │   └────┬─────┘       │
                       │        │             │
                       │        │             ▼
                       │        │      ┌──────────┐
                       │        │      │ Attempts │
                       │        │      │  < Max?  │
                       │        │      └──┬────┬──┘
                       │        │        Yes   No
                       │        │         │     │
                       │        │         │     ▼
                       │        │         │  ┌──────────┐
                       │        │         │  │ Fallback │
                       │        │         │  │   Node   │
                       │        │         │  └────┬─────┘
                       │        │         │       │
                       └────────┴─────────┴───────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │  Save Checkpoint │
                              │  to PostgreSQL   │
                              └─────────────────┘
```

## Features

### 1. Multi-Node Agent Graph

- **Agent Node**: LLM decides whether to use tools or provide a final answer
- **Tool Node**: Executes mathematical operations (add, multiply, divide)
- **Validation Node**: Checks answer quality (not empty, not a refusal, sufficient length)
- **Refine Node**: Attempts to improve invalid answers (up to MAX_ATTEMPTS)
- **Fallback Node**: Last-resort direct answer if refinement fails
- **Memory Extraction Node**: Automatically extracts and saves user memories

### 2. Long-Term Memory System

- **Automatic Extraction**: After each conversation, the LLM identifies personal info, preferences, and context
- **User-Specific Storage**: Memories are tied to a `user_id` and persist across conversations
- **Smart Loading**: Memories are automatically loaded and injected into the system prompt
- **Categories**: Organizes memories into `personal`, `preference`, `context`, and `fact`
- **Importance Scoring**: Each memory has an importance score (0.0-1.0) for prioritization

### 3. Checkpoint System

- **PostgreSQL Persistence**: Every conversation step is saved to PostgreSQL
- **Recovery**: If a run fails mid-execution, it can resume from the last checkpoint
- **Thread Isolation**: Each conversation gets a unique `thread_id` for isolation
- **State Snapshots**: Full state (messages, attempts, validation errors) is captured

### 4. Retry Logic

- **Exponential Backoff**: Retries with increasing delays (2s, 4s, 8s, ...)
- **Jitter**: Random jitter prevents thundering herd problems
- **Selective Retries**: Only retries transient errors (429, 5xx, timeouts, overloads)
- **Request Tracking**: Every attempt (including retries) is counted in LLM stats

### 5. Observability

- **LLM Stats**: Tracks requests, failures, prompt tokens, completion tokens
- **Per-Model Breakdown**: See which models consumed how many requests
- **Trace Events**: Full execution trace with tool calls, validations, and retries
- **Usage Endpoint**: Real-time view of cumulative LLM usage

## Project Flow

### Conversation Flow

1. **User sends a question** via `/chat` endpoint with optional `user_id`
2. **Load memories**: If `user_id` is provided, fetch all memories for that user
3. **Inject context**: Memories are added to the system prompt
4. **Agent processes**: LLM decides to use tools or answer directly
5. **Tool execution**: If tools are needed, they execute and return results
6. **Validation**: Answer is checked for quality (not empty, not a refusal, etc.)
7. **Refinement**: If invalid, the agent attempts to improve the answer (up to 2 times)
8. **Fallback**: If refinement fails, a direct answer is generated
9. **Memory extraction**: LLM analyzes the conversation and extracts personal info
10. **Save memories**: Extracted memories are stored in the database
11. **Checkpoint**: Every step is saved to PostgreSQL for recovery
12. **Return response**: Answer, trace, and LLM stats are returned to the user

### Memory Extraction Flow

After a successful conversation:

1. **Analyze conversation**: LLM reviews the full message history
2. **Identify memories**: Extracts personal info, preferences, context, and facts
3. **Categorize**: Each memory is tagged with a category and importance score
4. **Save to database**: Memories are stored with `user_id`, `category`, `key`, `value`
5. **Deduplicate**: If a memory with the same `(user_id, category, key)` exists, it's updated

### Checkpoint Recovery Flow

If a run fails mid-execution:

1. **Detect failure**: Exception is caught in `run_agent_with_recovery()`
2. **Load checkpoint**: Retrieve the last saved state from PostgreSQL
3. **Identify pending nodes**: Determine which nodes haven't completed
4. **Resume execution**: Continue from the last good checkpoint
5. **Retry**: Attempt up to `max_resumes` times (default: 3)

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL 12+
- OpenRouter API key

### Installation

1. **Clone the repository**:
   ```bash
   cd langchain_ai_assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install langchain-core langchain-openrouter langgraph langgraph-checkpoint-postgres
   pip install psycopg[binary] psycopg_pool
   pip install fastapi uvicorn python-dotenv pydantic
   ```

3. **Configure environment**:
   Create a `.env` file:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   POSTGRES_URI=postgresql://user:password@localhost:5432/dbname?sslmode=disable
   OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free  # optional
   ```

4. **Initialize database**:
   The memory and checkpoint tables are created automatically on first run.

5. **Start the API server**:
   ```bash
   python api.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Endpoints

### Chat Endpoints

#### POST `/chat`
Send a question and get a response.

**Request**:
```json
{
  "question": "What is (25 * 10) + 17, and what is that divided by 3?",
  "user_id": "user-123"
}
```

**Response**:
```json
{
  "question": "What is (25 * 10) + 17, and what is that divided by 3?",
  "answer": "The result is 89.",
  "trace": [
    {
      "node": "agent",
      "event": "tool_calls",
      "tool_calls": [
        {"name": "multiply_numbers", "args": {"a": 25, "b": 10}}
      ]
    },
    {
      "node": "tools",
      "event": "answer",
      "content": "250"
    }
  ],
  "llm": {
    "requests": 4,
    "failures": 0,
    "prompt_tokens": 1234,
    "completion_tokens": 567
  }
}
```

#### POST `/chat/stream`
Stream the response in real-time using Server-Sent Events.

**Request**: Same as `/chat`

**Response**: Stream of events:
```
data: {"node": "agent", "event": "tool_calls", "tool_calls": [...]}
data: {"node": "tools", "event": "answer", "content": "250"}
data: {"node": "validate_answer", "event": "validation", "status": "VALID"}
data: [DONE]
```

### Memory Endpoints

#### GET `/memories/{user_id}`
Get all memories for a specific user.

**Response**:
```json
{
  "user_id": "user-123",
  "count": 3,
  "memories": [
    {
      "memory_id": "uuid-here",
      "category": "personal",
      "key": "name",
      "value": "Alice",
      "importance": 0.9,
      "last_accessed": "2026-10-07T23:57:02.627971+05:30",
      "created_at": "2026-10-07T23:54:02.730903+05:30"
    }
  ]
}
```

#### POST `/memories`
Manually add a memory for a user.

**Request**:
```json
{
  "user_id": "user-123",
  "category": "preference",
  "key": "favorite_language",
  "value": "Python",
  "importance": 0.8
}
```

**Response**:
```json
{
  "status": "saved",
  "memory_id": "uuid-here",
  "user_id": "user-123",
  "category": "preference",
  "key": "favorite_language"
}
```

#### DELETE `/memories/{user_id}/{memory_id}`
Delete a specific memory.

**Response**:
```json
{
  "status": "deleted",
  "user_id": "user-123",
  "memory_id": "uuid-here"
}
```

#### DELETE `/memories/{user_id}`
Delete all memories for a user.

**Response**:
```json
{
  "status": "deleted",
  "user_id": "user-123",
  "deleted_count": 5
}
```

#### GET `/user/{user_id}`
Get user profile information.

**Response**:
```json
{
  "user_id": "user-123",
  "name": null,
  "email": null,
  "preferences": {},
  "created_at": "2026-10-07T23:54:02.730903+05:30",
  "updated_at": "2026-10-07T23:57:02.627971+05:30"
}
```

### Utility Endpoints

#### GET `/health`
Check API health and configuration.

**Response**:
```json
{
  "status": "ok",
  "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
  "max_attempts": 2
}
```

#### GET `/usage`
Get cumulative LLM usage statistics.

**Response**:
```json
{
  "requests": 42,
  "failures": 3,
  "prompt_tokens": 15234,
  "completion_tokens": 8765,
  "per_model": {
    "nvidia/nemotron-3-ultra-550b-a55b:free": 42
  }
}
```

#### POST `/usage/reset`
Reset LLM usage counters.

**Response**: Same as `/usage` (with zeroed values)

#### GET `/graph`
Get the agent graph structure.

**Response**:
```json
{
  "nodes": ["agent", "tools", "validate_answer", "refine", "fallback", "extract_memory"],
  "edges": [...],
  "conditional_edges": [...]
}
```

## Memory System

### How It Works

1. **Automatic Extraction**: After each conversation, the `extract_memory_node` runs
2. **LLM Analysis**: The LLM analyzes the conversation and identifies:
   - Personal information (name, age, location)
   - Preferences (favorite tools, languages, styles)
   - Context (job role, projects, goals)
   - Facts (any other relevant information)
3. **Structured Output**: Memories are returned as JSON with category, key, value, and importance
4. **Database Storage**: Memories are saved to the `memories` table with `user_id` as the key
5. **Deduplication**: If a memory with the same `(user_id, category, key)` exists, it's updated
6. **Context Injection**: Before the next conversation, memories are loaded and injected into the system prompt

### Memory Categories

- **personal**: Name, age, location, family, etc.
- **preference**: Likes, dislikes, preferred tools/languages, etc.
- **context**: Job role, projects, goals, etc.
- **fact**: Any other relevant facts about the user

### Importance Scoring

Each memory has an importance score (0.0-1.0):
- **0.9-1.0**: Critical information (name, core preferences)
- **0.7-0.8**: Important context (job role, major projects)
- **0.5-0.6**: Useful but not critical (minor preferences)
- **0.0-0.4**: Low priority (ephemeral details)

### Example Flow

**Conversation 1**:
```
User: "I'm a data scientist named Alice and I prefer Python for ML work"
Agent: "Hello Alice! ..."
→ Extracted memories:
  - [personal] name: Alice (0.9)
  - [context] job_role: data scientist (0.9)
  - [preference] ml_language: Python (0.9)
```

**Conversation 2**:
```
User: "What programming language should I use for my next project?"
→ Memories loaded: Alice, data scientist, Python
Agent: "Based on your context as a data scientist named Alice with a 
        preference for Python in machine learning, I'd recommend Python..."
```

## Checkpoint System

### How It Works

1. **PostgreSQL Storage**: Every conversation step is saved to PostgreSQL
2. **Linked List Structure**: Checkpoints form a linked list (ID → parent ID)
3. **State Snapshots**: Full state (messages, attempts, validation errors) is captured
4. **Thread Isolation**: Each conversation gets a unique `thread_id`
5. **Recovery**: If a run fails, it can resume from the last checkpoint

### Checkpoint Tables

- **checkpoints**: Main checkpoint data (ID, parent, step, state)
- **checkpoint_blobs**: Large binary data (serialized messages)
- **checkpoint_writes**: Pending writes (tool results not yet applied)
- **checkpoint_migrations**: Schema version tracking

### Recovery Example

```python
# Run with recovery
result = run_agent_with_recovery(
    question="Complex question...",
    thread_id="run-abc123",
    user_id="user-123",
    max_resumes=3
)

# If the run fails at step 5, it will:
# 1. Catch the exception
# 2. Load checkpoint from step 4
# 3. Resume from step 5
# 4. Retry up to 3 times
```

## Configuration

### Environment Variables

- **OPENROUTER_API_KEY**: Your OpenRouter API key (required)
- **POSTGRES_URI**: PostgreSQL connection string (required)
- **OPENROUTER_MODEL**: Model to use (optional, default: `nvidia/nemotron-3-ultra-550b-a55b:free`)

### Code Configuration

In `agent_graph.py`:

```python
MAX_ATTEMPTS = 2          # Max refinement attempts before fallback
MAX_RETRIES = 4           # Max retries per model call
RETRY_BASE_DELAY = 2.0    # Initial retry delay (seconds)
RETRY_MAX_DELAY = 20.0    # Maximum retry delay (seconds)
```

## Usage Examples

### Basic Chat

```python
import requests

response = requests.post(
    "http://localhost:8000/chat",
    json={
        "question": "What is 25 * 10?",
        "user_id": "user-123"
    }
)

print(response.json())
```

### Streaming Chat

```python
import requests

response = requests.post(
    "http://localhost:8000/chat/stream",
    json={
        "question": "What is 25 * 10?",
        "user_id": "user-123"
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

### View User Memories

```python
import requests

response = requests.get("http://localhost:8000/memories/user-123")
print(response.json())
```

### Manually Add Memory

```python
import requests

response = requests.post(
    "http://localhost:8000/memories",
    json={
        "user_id": "user-123",
        "category": "preference",
        "key": "favorite_color",
        "value": "blue",
        "importance": 0.7
    }
)
```

### Check LLM Usage

```python
import requests

response = requests.get("http://localhost:8000/usage")
print(response.json())
```

## Database Schema

### Users Table

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name TEXT,
    email TEXT,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

### Memories Table

```sql
CREATE TABLE memories (
    memory_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    importance FLOAT DEFAULT 0.5,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, category, key)
);
```

## Cost Considerations

### LLM Requests

- **Simple question**: ~1-2 requests (agent + validation)
- **Tool-based question**: ~3-4 requests (agent + tool + agent + validation)
- **With refinement**: +2 requests per attempt (up to 2 attempts)
- **Memory extraction**: +1 request per conversation
- **Retries**: Each retry counts as a request

### OpenRouter Free Tier

- **Daily limit**: 50 requests/day (account-wide, not per-model)
- **RPM limit**: 20 requests/minute
- **Upgrade**: $10 credit raises daily cap to 1,000 requests

### Optimization Tips

1. **Use a single model**: Avoid fallback chains that multiply requests
2. **Monitor usage**: Check `/usage` endpoint regularly
3. **Batch operations**: Combine multiple questions when possible
4. **Cache responses**: Store frequently-accessed answers
5. **Upgrade if needed**: $10 credit provides 20x more daily requests

## Testing

Run the memory system test:

```bash
python test_memory.py
```

This will:
1. Initialize memory tables
2. Run a conversation with memory extraction
3. Verify memories are saved
4. Run a second conversation to verify memories are loaded

## Troubleshooting

### Memory Extraction Not Working

- Check that `user_id` is provided in the request
- Verify PostgreSQL connection is working
- Check logs for `[memory]` messages

### Checkpoint Recovery Failing

- Verify `POSTGRES_URI` is set correctly
- Check that PostgreSQL is running
- Ensure the user has permissions to create tables

### High Retry Rate

- Check OpenRouter status page for outages
- Consider switching to a less overloaded model
- Monitor `/usage` endpoint for failure counts

### Rate Limiting (429 Errors)

- You've hit the 20 RPM or 50 requests/day limit
- Wait a minute or upgrade to a paid plan
- Check `/usage` endpoint to see current usage

## License

This project is for educational and personal use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please open an issue on GitHub.
