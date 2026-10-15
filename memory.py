import os
import uuid
from datetime import datetime
from typing import Optional

import numpy as np
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer


_embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model


def generate_embedding(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text)
    return embedding.tolist()


_memory_pool: ConnectionPool | None = None


def get_memory_pool() -> ConnectionPool:
    global _memory_pool
    if _memory_pool is None:
        uri = os.getenv("POSTGRES_URI")
        if not uri:
            raise RuntimeError(
                "POSTGRES_URI is not set. Add it to your .env file."
            )
        _memory_pool = ConnectionPool(
            conninfo=uri,
            max_size=10,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=True,
        )
    return _memory_pool


def init_memory_tables() -> None:
    pool = get_memory_pool()
    with pool.connection() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                email TEXT,
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                memory_id UUID PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                importance FLOAT DEFAULT 0.5,
                embedding vector(384),
                last_accessed TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(user_id, category, key)
            )
        """)
        conn.execute("""
            ALTER TABLE memories
            ADD COLUMN IF NOT EXISTS embedding vector(384)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_user_id
            ON memories(user_id)
        """)
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding
                ON memories USING hnsw (embedding vector_cosine_ops)
            """)
        except Exception as exc:
            print(f"[memory] vector index creation skipped: {exc}")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS user_threads (
                user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                thread_id TEXT NOT NULL,
                title TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_active TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (user_id, thread_id)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_threads_user_id
            ON user_threads(user_id)
        """)


def ensure_user(user_id: str, name: Optional[str] = None, email: Optional[str] = None) -> None:
    pool = get_memory_pool()
    with pool.connection() as conn:
        conn.execute("""
            INSERT INTO users (user_id, name, email)
            VALUES (%s, %s, %s)
            ON CONFLICT (user_id) DO UPDATE
            SET name = COALESCE(EXCLUDED.name, users.name),
                email = COALESCE(EXCLUDED.email, users.email),
                updated_at = NOW()
        """, (user_id, name, email))


def save_memory(user_id: str, category: str, key: str, value: str, importance: float = 0.5) -> str:
    ensure_user(user_id)
    memory_id = str(uuid.uuid4())
    memory_text = f"{category}: {key} = {value}"
    embedding = generate_embedding(memory_text)

    pool = get_memory_pool()
    with pool.connection() as conn:
        register_vector(conn)
        conn.execute("""
            INSERT INTO memories (memory_id, user_id, category, key, value, importance, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, category, key) DO UPDATE
            SET value = EXCLUDED.value,
                importance = EXCLUDED.importance,
                embedding = EXCLUDED.embedding,
                last_accessed = NOW()
        """, (memory_id, user_id, category, key, value, importance, embedding))
    return memory_id


def get_user_memories(user_id: str) -> list[dict]:
    pool = get_memory_pool()
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT memory_id, category, key, value, importance, last_accessed, created_at
            FROM memories
            WHERE user_id = %s
            ORDER BY importance DESC, last_accessed DESC
        """, (user_id,)).fetchall()

        with pool.connection() as update_conn:
            update_conn.execute("""
                UPDATE memories SET last_accessed = NOW()
                WHERE user_id = %s
            """, (user_id,))

    return [
        {
            "memory_id": str(row[0]),
            "category": row[1],
            "key": row[2],
            "value": row[3],
            "importance": row[4],
            "last_accessed": row[5].isoformat() if row[5] else None,
            "created_at": row[6].isoformat() if row[6] else None,
        }
        for row in rows
    ]


def search_memories(user_id: str, query: str, limit: int = 10) -> list[dict]:
    pool = get_memory_pool()
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT memory_id, category, key, value, importance, last_accessed, created_at
            FROM memories
            WHERE user_id = %s
              AND (key ILIKE %s OR value ILIKE %s OR category ILIKE %s)
            ORDER BY importance DESC, last_accessed DESC
            LIMIT %s
        """, (user_id, f"%{query}%", f"%{query}%", f"%{query}%", limit))

    return [
        {
            "memory_id": str(row[0]),
            "category": row[1],
            "key": row[2],
            "value": row[3],
            "importance": row[4],
            "last_accessed": row[5].isoformat() if row[5] else None,
            "created_at": row[6].isoformat() if row[6] else None,
        }
        for row in rows
    ]


def semantic_search(user_id: str, query: str, top_k: int = 3) -> list[dict]:
    """Semantic search using vector similarity."""
    query_embedding = generate_embedding(query)

    pool = get_memory_pool()
    with pool.connection() as conn:
        register_vector(conn)
        rows = conn.execute("""
            SELECT memory_id, category, key, value, importance,
                   1 - (embedding <=> %s::vector) as similarity,
                   last_accessed, created_at
            FROM memories
            WHERE user_id = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, user_id, query_embedding, top_k)).fetchall()

    return [
        {
            "memory_id": str(row[0]),
            "category": row[1],
            "key": row[2],
            "value": row[3],
            "importance": row[4],
            "similarity": float(row[5]),
            "last_accessed": row[6].isoformat() if row[6] else None,
            "created_at": row[7].isoformat() if row[7] else None,
        }
        for row in rows
    ]


def delete_memory(user_id: str, memory_id: str) -> bool:
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            DELETE FROM memories
            WHERE user_id = %s AND memory_id = %s
        """, (user_id, memory_id))
        return result.rowcount > 0


def delete_all_user_memories(user_id: str) -> int:
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            DELETE FROM memories WHERE user_id = %s
        """, (user_id,))
        return result.rowcount


def get_user_profile(user_id: str) -> Optional[dict]:
    pool = get_memory_pool()
    with pool.connection() as conn:
        row = conn.execute("""
            SELECT user_id, name, email, preferences, created_at, updated_at
            FROM users
            WHERE user_id = %s
        """, (user_id,)).fetchone()

    if not row:
        return None

    return {
        "user_id": row[0],
        "name": row[1],
        "email": row[2],
        "preferences": row[3],
        "created_at": row[4].isoformat() if row[4] else None,
        "updated_at": row[5].isoformat() if row[5] else None,
    }


def format_memories_for_prompt(memories: list[dict]) -> str:
    if not memories:
        return ""

    lines = ["User context from previous interactions:"]
    for mem in memories:
        lines.append(f"- {mem['category']}: {mem['key']} = {mem['value']}")

    return "\n".join(lines)


def register_thread(user_id: str, thread_id: str, title: str | None = None) -> None:
    """Register a conversation thread for a user."""
    if not user_id or not thread_id:
        return

    ensure_user(user_id)
    pool = get_memory_pool()
    with pool.connection() as conn:
        conn.execute("""
            INSERT INTO user_threads (user_id, thread_id, title, last_active)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (user_id, thread_id) DO UPDATE
            SET last_active = NOW(),
                title = COALESCE(EXCLUDED.title, user_threads.title)
        """, (user_id, thread_id, title))


def get_user_threads(user_id: str, limit: int = 50) -> list[dict]:
    """Get all conversation threads for a user."""
    pool = get_memory_pool()
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT thread_id, title, created_at, last_active
            FROM user_threads
            WHERE user_id = %s
            ORDER BY last_active DESC
            LIMIT %s
        """, (user_id, limit)).fetchall()

    return [
        {
            "thread_id": row[0],
            "title": row[1],
            "created_at": row[2].isoformat() if row[2] else None,
            "last_active": row[3].isoformat() if row[3] else None,
        }
        for row in rows
    ]


def delete_thread(user_id: str, thread_id: str) -> bool:
    """Delete a thread mapping for a user."""
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            DELETE FROM user_threads
            WHERE user_id = %s AND thread_id = %s
        """, (user_id, thread_id))
        return result.rowcount > 0
