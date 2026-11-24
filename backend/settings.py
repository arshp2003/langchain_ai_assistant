"""Settings read from the project .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _int_setting(name: str, default: int) -> int:
    """Read an integer setting, treating a blank .env entry as unset."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None


# Chunking parameters live in one place so the upload endpoint cannot drift
# from whatever else chunks a document later.
CHUNK_SIZE = _int_setting("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _int_setting("CHUNK_OVERLAP", 100)

if CHUNK_SIZE < 1:
    raise ValueError(f"CHUNK_SIZE must be positive, got {CHUNK_SIZE}")
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        f"CHUNK_OVERLAP must be between 0 and CHUNK_SIZE ({CHUNK_SIZE}), "
        f"got {CHUNK_OVERLAP}"
    )
