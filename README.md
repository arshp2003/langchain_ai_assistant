# LangChain AI Assistant with Long-Term Memory

A production-ready AI assistant built with LangGraph that combines short-term conversation persistence (checkpoints) with long-term user memory, advanced RAG pipeline, and document management.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [RAG Pipeline](#rag-pipeline)
- [Project Flow](#project-flow)
- [Setup](#setup)
- [API Endpoints](#api-endpoints)
- [Memory System](#memory-system)
- [Checkpoint System](#checkpoint-system)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Testing](#testing)

## Overview

This project implements a multi-node ReAct agent with:

- **Short-term memory**: PostgreSQL checkpointing saves every conversation step, enabling recovery from failures
- **Long-term memory**: Automatic extraction and storage of user preferences, personal info, and context across conversations
- **Advanced RAG**: 5-step retrieval pipeline (rerank → filter → trim → compress → validate) for high-quality document answers
- **Document management**: Upload, search, and download documents with vector embedding storage
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
                       │   ┌──────────┐  ┌──────────
                       │   │ Extract  │  │  Refine  │
                       │   │ Memory   │  │   Node   │
                       │   │  Node    │  └────┬─────┘
                       │   └────┬─────┘       │
                       │        │             │
                       │        │             ▼
                       │        │      ┌──────────┐
                       │        │      │ Attempts │
                       │        │      │  < Max?  │
                       │        │      └──────┬──┘
                       │        │        Yes   No
                       │        │         │     │
                       │        │         │     ▼
                       │        │         │  ┌──────────┐
                       │        │         │  │ Fallback │
                       │        │         │  │   Node   │
                       │        │         │  └────┬─────
                       │        │         │       │
                       └────────┴─────────┴───────┘
                                        │
                                        ▼
                              ─────────────────┐
                              │  Save Checkpoint │
                              │  to PostgreSQL   │
                              └─────────────────┘
```

## Features

### 1. Multi-Node Agent Graph

- **Agent Node**: LLM decides whether to use tools or provide a final answer
- **Tool Node**: Executes mathematical operations (add, multiply, divide) and document search
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
- **Soft Delete**: Memories use a deleted flag + timestamp instead of hard delete; can be restored

### 3. Advanced RAG Pipeline

A 5-step retrieval pipeline that transforms raw vector search results into high-quality, concise context:

1. **Cross-Encoder Reranking**: Re-scores vector search results using a local cross-encoder model (`ms-marco-MiniLM-L-6-v2`) for true relevance
2. **Relevance Filtering**: Drops chunks below a score threshold (1.0) to prevent irrelevant context
3. **Passage Trimming**: Keeps only query-relevant sentences within each chunk using relative margin scoring
4. **Context Compression**: LLM rewrites verbose context into shorter form (skipped for short contexts <600 tokens)
5. **Validation**: Embedding similarity check ensures compression didn't lose answer-relevant information

**Performance**: Typically achieves 70-80% context reduction while preserving all key facts.

### 4. Document Management

- **Upload**: PDF, TXT, and MD files are automatically chunked and embedded into ChromaDB
- **Search**: Semantic search across all uploaded documents with the full RAG pipeline
- **Download**: Generate clickable download links for any uploaded document
- **List**: View all uploaded documents with metadata

### 5. Checkpoint System

- **PostgreSQL Persistence**: Every conversation step is saved to PostgreSQL
- **Recovery**: If a run fails mid-execution, it can resume from the last checkpoint
- **Thread Isolation**: Each conversation gets a unique `thread_id` for isolation
- **State Snapshots**: Full state (messages, attempts, validation errors) is captured

### 6. Retry Logic

- **Exponential Backoff**: Retries with increasing delays (2s, 4s, 8s, ...)
- **Jitter**: Random jitter prevents thundering herd problems
- **Selective Retries**: Only retries transient errors (429, 5xx, timeouts, overloads)
- **Request Tracking**: Every attempt (including retries) is counted in LLM stats

### 7. Observability

- **LLM Stats**: Tracks requests, failures, prompt tokens, completion tokens
- **Per-Model Breakdown**: See which models consumed how many requests
- **Trace Events**: Full execution trace with tool calls, validations, and retries
- **Usage Endpoint**: Real-time view of cumulative LLM usage

### 8. Human-in-the-Loop (HITL)

- **Tool Approval**: High-risk tools (delete operations) require human approval before execution
- **Auto-Approval**: Low-risk tools (search, list, download, weather lookup) execute automatically
- **Memory Approval**: Memory extraction is auto-approved but can be reviewed
- **Rejection Handling**: Rejected actions inject feedback for the agent to adjust

## RAG Pipeline

### How It Works

```
User Query
    │
    ▼
┌─────────────────────┐
│  Vector Search (k=10)│  ← Fast semantic search via ChromaDB
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Cross-Encoder       │  ← Re-scores by true relevance (~100ms)
│  Rerank (top 3)      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Relevance Filter    │  ← Drop chunks with score < 1.0
│  (threshold=1.0)     │
──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Passage Trimming    │  ← Keep only relevant sentences per chunk
│  (margin=1.0, cap=4) │     (67% reduction typical)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Context Compression │  ← LLM rewrites if >600 tokens
│  (skip if short)     │     (54% reduction typical)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Validation          │  ← Check similarity drop < 0.15
│  (fallback if bad)   │     (embedding-based, ~10ms)
└──────────┬──────────┘
           │
           ▼
    Final Context
    (70-80% smaller,
     all facts preserved)
```

### Key Design Decisions

- **Local cross-encoder** over API-based rerankers: No rate limits, works offline, already installed
- **Relative margin for trimming**: Cross-encoder absolute scores vary by text length; ranking is stable
- **Token threshold for compression**: Skip LLM call when context is short (saves requests on free tier)
- **Embedding similarity for validation**: Cheap (~10ms) vs LLM-based validation (costs a request)

### Files

- `backend/reranker.py` — Cross-encoder reranking + relevance filtering
- `backend/trimmer.py` — Sentence-level passage trimming
- `backend/compressor.py` — LLM context compression with token threshold
- `backend/validator.py` — Embedding similarity validation with fallback

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

### Document Search Flow (RAG)

1. **User asks about document content**: "What does my PDF say about X?"
2. **Vector search**: Retrieve top 10 chunks from ChromaDB by embedding similarity
3. **Rerank**: Cross-encoder re-scores all 10 chunks, keeps top 3
4. **Filter**: Drop any chunks with relevance score < 1.0
5. **Trim**: For each chunk, keep only sentences within margin of the best sentence
6. **Compress**: If total context > 600 tokens, LLM rewrites to shorter form
7. **Validate**: Check that compressed context still has high similarity to query
8. **Answer**: LLM generates answer using the refined context

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
   pip install -r requirements.txt
   ```

3. **Configure environment**:
   Create a `.env` file:
   ```env
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   POSTGRES_URI=postgresql://user:password@localhost:5432/dbname?sslmode=disable
   OPENROUTER_MODEL=nvidia/nemotron-3.5-super-vl:free  # optional
   OPENWEATHER_API_KEY=your-key-here                # optional, for get_weather
   CHUNK_SIZE=500                                     # optional
   CHUNK_OVERLAP=100                                  # optional
   ```

4. **Initialize database**:
   The memory and checkpoint tables are created automatically on first run.

5. **Start the API server**:
   ```bash
   cd backend
   python api.py
   ```
   
   Or with uvicorn:
   ```bash
   cd backend
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Open the frontend**:
   Open `frontend/index.html` in your browser.

## API Endpoints

### Chat Endpoints

#### POST `/chat`
Send a question and get a response.

**Request**:
```json
{
  "question": "What is (25 * 10) + 17, and what is that divided by 3?",
  "user_id": "user-123",
  "thread_id": null
}
```

Send `thread_id` back from the previous response to make this question another
turn in the same conversation. Leave it `null` (or omit it) to start a new
thread with no history.

**Response**:
```json
{
  "question": "What is (25 * 10) + 17, and what is that divided by 3?",
  "answer": "The result is 89.",
  "thread_id": "run-9e7a8ecc68584376b0f97e9572f31bc8",
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

#### POST `/chat/hitl`
Human-in-the-loop mode. Pauses before tool execution and memory extraction for approval.

**Request**: Same as `/chat`

**Response**: Includes `pending_approval` field if action requires review.

### Document Endpoints

#### POST `/documents/upload`
Upload a document (PDF, TXT, or MD) for RAG-powered search.

**Request**: Multipart form data with `file` field

**Response**:
```json
{
  "status": "uploaded",
  "filename": "report.pdf",
  "chunks_created": 15,
  "vectors_stored": 15
}
```

#### GET `/documents`
List all uploaded documents.

**Response**:
```json
{
  "documents": [
    {
      "filename": "report.pdf",
      "size_bytes": 387912,
      "type": "PDF"
    }
  ],
  "count": 1
}
```

#### GET `/documents/{filename}/download`
Download an uploaded document.

**Response**: File download with appropriate MIME type.

#### DELETE `/documents/{filename}`
Delete a document file (vector embeddings may remain in ChromaDB).

**Response**:
```json
{
  "status": "deleted",
  "filename": "report.pdf",
  "note": "File removed. Vector embeddings may still exist in ChromaDB."
}
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
Soft-delete a specific memory (can be restored).

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

#### GET `/threads/{user_id}`
Get all conversation threads for a user (like ChatGPT's conversation history).

**Query Parameters**:
- `limit` (optional): Maximum number of threads to return (default: 50)

**Response**:
```json
{
  "user_id": "user-123",
  "count": 3,
  "threads": [
    {
      "thread_id": "run-9e7a8ecc68584376b0f97e9572f31bc8",
      "title": "What is 10 + 20?",
      "created_at": "2026-10-08T23:22:05.793557+05:30",
      "last_active": "2026-10-08T23:22:05.793557+05:30"
    }
  ]
}
```

#### DELETE `/threads/{user_id}/{thread_id}`
Delete a thread mapping for a user.

**Response**:
```json
{
  "status": "deleted",
  "user_id": "user-123",
  "thread_id": "run-9e7a8ecc68584376b0f97e9572f31bc8"
}
```

### Approval Endpoints (HITL)

#### GET `/approval/{thread_id}`
Check what action the graph is waiting to execute next.

**Response**:
```json
{
  "status": "awaiting_approval",
  "thread_id": "run-abc123",
  "pending_node": "tools",
  "action": "execute_tools",
  "tool_calls": [
    {"name": "delete_document", "args": {"filename": "report.pdf"}}
  ]
}
```

#### POST `/approval/{thread_id}`
Approve or reject the pending action and resume the graph.

**Request**:
```json
{
  "approved": true,
  "feedback": ""
}
```

**Response**:
```json
{
  "status": "resumed",
  "approved": true,
  "thread_id": "run-abc123",
  "answer": "Document deleted successfully.",
  "trace": [...],
  "still_pending": false
}
```

### Utility Endpoints

#### GET `/health`
Check API health and configuration.

**Response**:
```json
{
  "status": "ok",
  "model": "nvidia/nemotron-3.5-super-vl:free",
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
    "nvidia/nemotron-3.5-super-vl:free": 42
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

### Conversation History (ChatGPT-like)

The system now supports persistent conversation history, similar to ChatGPT:

1. **Automatic Thread Registration**: Every conversation is automatically registered with the user's thread list
2. **Thread Listing**: Use `GET /threads/{user_id}` to list all conversations for a user
3. **Resume Conversations**: Load a previous thread and continue the conversation
4. **Thread Metadata**: Each thread has a title (first 50 chars of the question), created_at, and last_active timestamp

**Example Flow**:
```python
# Day 1: Have a conversation
result = run_agent(
    "I'm learning LangGraph",
    thread_id="my-chat-123",
    user_id="arsh"
)
# Thread is automatically registered

# Day 2: List all conversations
threads = get_user_threads("arsh")
# Returns: [{"thread_id": "my-chat-123", "title": "I'm learning LangGraph", ...}]

# Continue the conversation
result = run_agent(
    "Can you explain checkpoints?",
    thread_id="my-chat-123",  # Same thread_id!
    user_id="arsh"
)
# Agent sees the full history and continues naturally
```

### Long-Term Memory

#### How It Works

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

#### Memory Categories

- **personal**: Name, age, location, family, etc.
- **preference**: Likes, dislikes, preferred tools/languages, etc.
- **context**: Job role, projects, goals, etc.
- **fact**: Any other relevant facts about the user

#### Importance Scoring

Each memory has an importance score (0.0-1.0):
- **0.9-1.0**: Critical information (name, core preferences)
- **0.7-0.8**: Important context (job role, major projects)
- **0.5-0.6**: Useful but not critical (minor preferences)
- **0.0-0.4**: Low priority (ephemeral details)

#### Example Flow

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
- **OPENROUTER_MODEL**: Model to use (optional, default: `nvidia/nemotron-3.5-super-vl:free`)
- **OPENWEATHER_API_KEY**: Free key from openweathermap.org, used by the `get_weather` tool (optional: without it the tool explains that weather lookup is not configured)
- **CHUNK_SIZE**: Characters per document chunk at ingestion (optional, default: `500`)
- **CHUNK_OVERLAP**: Characters repeated between consecutive chunks (optional, default: `100`, must be between `0` and `CHUNK_SIZE`)

### Code Configuration

Chunking is the only setting routed through `settings.py` (`CHUNK_SIZE`, `CHUNK_OVERLAP`), which the upload endpoint reads so no ingestion path can drift from another. The model slug, the weather key and the database URI are each read in the module that uses them. Everything else below is set in code:

In `agent_graph.py`:

```python
MAX_ATTEMPTS = 2          # Max refinement attempts before fallback
MAX_RETRIES = 4           # Max retries per model call
RETRY_BASE_DELAY = 2.0    # Initial retry delay (seconds)
RETRY_MAX_DELAY = 20.0    # Maximum retry delay (seconds)
```

In `trimmer.py`:

```python
DEFAULT_MARGIN = 1.0      # Keep sentences within this score of the best
DEFAULT_MAX_SENTENCES = 4 # Hard cap on sentences kept per chunk
```

In `compressor.py`:

```python
TOKEN_THRESHOLD = 600     # Skip compression if context is shorter than this
```

In `validator.py`:

```python
MAX_SIMILARITY_DROP = 0.15  # Fall back to trimmed if compression drops similarity more than this
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

### Upload and Search Documents

```python
import requests

# Upload a document
with open("report.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/documents/upload",
        files={"file": f}
    )
print(response.json())

# Ask about the document
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "question": "What does the report say about Q3 revenue?",
        "user_id": "user-123"
    }
)
print(response.json())
```

### Download a Document

```python
import requests
import webbrowser

# Get download link
response = requests.get("http://localhost:8000/documents/report.pdf/download")

# Open in browser (or save to file)
webbrowser.open("http://localhost:8000/documents/report.pdf/download")
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
    deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMPTZ,
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

All tests are in `backend/tests/`:

```bash
cd backend/tests

# Test individual components
python test_reranker.py      # Cross-encoder reranking + filtering
python test_trimmer.py       # Passage trimming (67% reduction)
python test_compressor.py    # LLM compression (54% reduction)
python test_validator.py     # Similarity validation + fallback
python test_download.py      # Document download endpoint + tool

# Test full pipeline end-to-end
python test_full_pipeline.py # All 5 RAG steps together (76% reduction)

# Test memory system
python test_memory.py        # Long-term memory extraction + loading
```

Each test file is self-contained and can be run independently.

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

### RAG Returns No Results

- Verify documents are uploaded (`GET /documents`)
- Check that ChromaDB directory exists (`backend/chroma_db`)
- Try a different query that matches the document content

## License

This project is for educational and personal use.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For issues or questions, please open an issue on GitHub.
