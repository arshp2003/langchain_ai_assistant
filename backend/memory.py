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


# -----------------------------
# Schema history (migrations)
# -----------------------------
#
# One entry per change to these tables, in the order the change was introduced,
# and the position in the list IS the version number. A database records the
# versions it has reached in `schema_migrations`, so a change runs once per
# database instead of being re-attempted on every boot, and "which schema is
# this database on?" becomes a question you can answer with a query.
#
# Append only. Never edit or reorder an existing entry: a database that already
# recorded version 4 would then be holding a different statement under that
# number, and nothing would notice.
#
# The second item of each pair asks "is this change already true of this
# database?". It is consulted only while the ledger is still empty, which is how
# a database built by the old re-run-everything code adopts the schema it already
# has instead of being migrated blind - while a database genuinely missing one
# change still gets exactly that one.
#
# A probe must return zero rows when the change is absent. `SELECT <expr> IS
# NOT NULL` always returns one row, and a row holding false is still a row, so
# it would report an absent table as present and skip the CREATE that defines it.
MIGRATIONS = [
    (
        "CREATE EXTENSION IF NOT EXISTS vector",
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'",
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            preferences JSONB DEFAULT '{}',
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
        """,
        "SELECT 1 WHERE to_regclass('public.users') IS NOT NULL",
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS memories (
            memory_id UUID PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            category TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            importance FLOAT DEFAULT 0.5,
            embedding vector(384),
            deleted BOOLEAN DEFAULT FALSE,
            deleted_at TIMESTAMPTZ,
            last_accessed TIMESTAMPTZ DEFAULT NOW(),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(user_id, category, key)
        )
        """,
        "SELECT 1 WHERE to_regclass('public.memories') IS NOT NULL",
    ),
    (
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS embedding vector(384)",
        """SELECT 1 FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = 'memories'
             AND column_name = 'embedding'""",
    ),
    (
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted BOOLEAN DEFAULT FALSE",
        """SELECT 1 FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = 'memories'
             AND column_name = 'deleted'""",
    ),
    (
        "ALTER TABLE memories ADD COLUMN IF NOT EXISTS deleted_at TIMESTAMPTZ",
        """SELECT 1 FROM information_schema.columns
           WHERE table_schema = 'public' AND table_name = 'memories'
             AND column_name = 'deleted_at'""",
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)",
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_memories_user_id'",
    ),
    (
        """
        CREATE TABLE IF NOT EXISTS user_threads (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            thread_id TEXT NOT NULL,
            title TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active TIMESTAMPTZ DEFAULT NOW(),
            PRIMARY KEY (user_id, thread_id)
        )
        """,
        "SELECT 1 WHERE to_regclass('public.user_threads') IS NOT NULL",
    ),
    (
        "CREATE INDEX IF NOT EXISTS idx_user_threads_user_id ON user_threads(user_id)",
        "SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_user_threads_user_id'",
    ),
]

# A fixed name in Postgres's cluster-wide advisory-lock space, so two processes
# booting at the same moment cannot both migrate. The number is arbitrary; it
# only has to stay the same.
_MIGRATION_LOCK_ID = 728391


def init_memory_tables() -> None:
    """Bring this database's schema up to the last entry in MIGRATIONS."""
    pool = get_memory_pool()
    with pool.connection() as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    v INTEGER PRIMARY KEY
                )
            """)
            applied = {
                row[0]
                for row in conn.execute("SELECT v FROM schema_migrations").fetchall()
            }
            adopt_existing = not applied

            for version, (sql, probe) in enumerate(MIGRATIONS):
                if version in applied:
                    continue
                if adopt_existing and conn.execute(probe).fetchone():
                    print(f"[memory] migration {version} already present, adopting")
                else:
                    conn.execute(sql)
                    print(f"[memory] migration {version} applied")
                conn.execute(
                    "INSERT INTO schema_migrations (v) VALUES (%s)", (version,)
                )

            _ensure_vector_index(conn)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))


def _ensure_vector_index(conn) -> None:
    """Best-effort HNSW index. Deliberately not a numbered migration: this is an
    optimisation, so a failure must neither block boot nor be recorded as
    applied. Recording something that did not happen is worse than recording
    nothing."""
    try:
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_embedding
            ON memories USING hnsw (embedding vector_cosine_ops)
        """)
    except Exception as exc:
        print(f"[memory] vector index creation skipped: {exc}")


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
        # RETURNING, not the minted uuid: on a conflict the row keeps its
        # original memory_id, so returning the new one hands callers an id that
        # exists nowhere - and their next delete by that id silently no-ops.
        stored_id = conn.execute("""
            INSERT INTO memories (memory_id, user_id, category, key, value, importance, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, category, key) DO UPDATE
            SET value = EXCLUDED.value,
                importance = EXCLUDED.importance,
                embedding = EXCLUDED.embedding,
                deleted = FALSE,
                deleted_at = NULL,
                last_accessed = NOW()
            RETURNING memory_id
        """, (memory_id, user_id, category, key, value, importance, embedding)).fetchone()[0]
    return str(stored_id)


def get_user_memories(user_id: str) -> list[dict]:
    """Every live memory for a user, highest importance first.

    Does not touch last_accessed - listing is not the same as using. See
    touch_memories(), which the prompt path calls for the rows it injects.
    """
    pool = get_memory_pool()
    with pool.connection() as conn:
        rows = conn.execute("""
            SELECT memory_id, category, key, value, importance, last_accessed, created_at
            FROM memories
            WHERE user_id = %s AND (deleted = FALSE OR deleted IS NULL)
            ORDER BY importance DESC, last_accessed DESC
        """, (user_id,)).fetchall()

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


def touch_memories(user_id: str, memory_ids: list[str]) -> int:
    """Mark the memories that were actually used as recently accessed.

    Called with the ids that went into a prompt, not with everything the user
    owns. Stamping every row on every read makes last_accessed identical across
    the store, which destroys the only signal for spotting stale memories.
    """
    if not memory_ids:
        return 0

    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            UPDATE memories SET last_accessed = NOW()
            WHERE user_id = %s AND memory_id = ANY(%s::uuid[])
        """, (user_id, list(memory_ids)))
        return result.rowcount


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
            WHERE user_id = %s AND (deleted = FALSE OR deleted IS NULL)
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
    """Soft delete a memory by marking it as deleted."""
    import uuid
    
    # Validate UUID format
    try:
        uuid.UUID(memory_id)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid memory_id format: '{memory_id}'. Expected a full UUID like 'e2704ab7-1234-5678-9abc-def012345678'")
    
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            UPDATE memories
            SET deleted = TRUE, deleted_at = NOW()
            WHERE user_id = %s AND memory_id = %s AND deleted = FALSE
        """, (user_id, memory_id))
        return result.rowcount > 0


def delete_all_user_memories(user_id: str) -> int:
    """Soft delete all memories for a user."""
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            UPDATE memories
            SET deleted = TRUE, deleted_at = NOW()
            WHERE user_id = %s AND (deleted = FALSE OR deleted IS NULL)
        """, (user_id,))
        return result.rowcount


def restore_memory(user_id: str, memory_id: str) -> bool:
    """Restore a soft-deleted memory."""
    import uuid
    
    try:
        uuid.UUID(memory_id)
    except (ValueError, AttributeError):
        raise ValueError(f"Invalid memory_id format: '{memory_id}'")
    
    pool = get_memory_pool()
    with pool.connection() as conn:
        result = conn.execute("""
            UPDATE memories
            SET deleted = FALSE, deleted_at = NULL
            WHERE user_id = %s AND memory_id = %s AND deleted = TRUE
        """, (user_id, memory_id))
        return result.rowcount > 0


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
