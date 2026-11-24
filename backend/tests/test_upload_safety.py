"""Upload safety: path traversal guard + duplicate/re-upload handling.

Runs against a temp documents dir and a temp Chroma dir so the real
backend/chroma_db is never written to.
"""
import asyncio
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
load_dotenv(BACKEND.parent / ".env")

import api

FAILS = []


def check(label, cond, extra=""):
    print(f"  [{'OK' if cond else 'FAIL'}] {label}{(' -> ' + extra) if extra else ''}")
    if not cond:
        FAILS.append(label)


class FakeUpload:
    def __init__(self, filename, content: bytes):
        self.filename = filename
        self._content = content

    async def read(self):
        return self._content


def post(filename, text):
    return asyncio.run(api.upload_document(FakeUpload(filename, text.encode())))


def stored_count():
    return api._get_vector_store()._collection.count()


tmp = Path(tempfile.mkdtemp(prefix="upload_safety_"))
api.DOCUMENTS_DIR = tmp / "documents"
api.CHROMA_DIR = tmp / "chroma"

try:
    print("\n=== 1. traversal payloads never escape the documents dir ===")
    payloads = [
        "../../outside.txt",
        "..\\..\\outside_windows.txt",
        "../../../outside_deep.txt",
        "/etc/passwd.txt",
        "sub/dir/nested.txt",
    ]
    for name in payloads:
        result = post(name, "some document text about retrieval augmented generation pipelines")
        flat = api.DOCUMENTS_DIR / Path(name.replace("\\", "/")).name
        check(f"{name!r:28s} written inside documents dir", flat.exists())

    strays = [
        p for p in tmp.rglob("*")
        if p.is_file()
        and api.DOCUMENTS_DIR not in p.parents
        and api.CHROMA_DIR not in p.parents
    ]
    check("no file written anywhere outside DOCUMENTS_DIR", not strays, str(strays))

    print("\n=== 2. rejected payloads ===")
    check("None filename", post(None, "text")["status"] == "error")
    check("empty filename", post("", "text")["status"] == "error")
    check("'..' as filename", post("..", "text")["status"] == "error")
    check("no extension (/etc/passwd)", post("/etc/passwd", "text")["status"] == "error")
    bad_type = post("notes.txt.exe", "text")
    check("disallowed extension", bad_type["status"] == "error", bad_type.get("error", ""))
    check("unsupported type never written to disk",
          not (api.DOCUMENTS_DIR / "notes.txt.exe").exists())

    print("\n=== 3. uploading the same file twice ===")
    body = "LangChain chains language model calls into a graph of runnable steps. " * 20
    first = post("notes.txt", body)
    check("first upload accepted", first["status"] == "uploaded", str(first))
    check("first upload not flagged duplicate", not first.get("already_uploaded"))
    check("first upload produced several chunks", first["chunks_created"] >= 3,
          f"chunks_created={first['chunks_created']}")
    count_after_first = stored_count()
    print(f"       chunks_created={first['chunks_created']} collection={count_after_first}")

    second = post("notes.txt", body)
    check("second identical upload flagged", second.get("already_uploaded") is True, str(second))
    check("second upload still reports status uploaded (frontend contract)",
          second["status"] == "uploaded")
    check("second upload returns a user-facing message", bool(second.get("message")),
          second.get("message", ""))
    check("collection did NOT grow on duplicate", stored_count() == count_after_first,
          f"{count_after_first} -> {stored_count()}")
    check("duplicate reported the real indexed chunk count",
          second["chunks_created"] == api._indexed_chunk_count("notes.txt"),
          str(second["chunks_created"]))

    print("\n=== 4. re-uploading a CHANGED file replaces it ===")
    changed = "LangGraph adds branching state, checkpointing and retries on top of chains. " * 30
    old_src = api._indexed_chunk_count("notes.txt")
    before = stored_count()
    third = post("notes.txt", changed)
    check("changed upload not flagged duplicate", not third.get("already_uploaded"))
    check("changed upload marked replaced", third.get("replaced") is True, str(third))
    check("changed upload carries a message", bool(third.get("message")), third.get("message", ""))
    check("changed upload produced more chunks than the old version",
          third["chunks_created"] > old_src, f"{old_src} -> {third['chunks_created']}")
    check("only the new version of this file is indexed",
          api._indexed_chunk_count("notes.txt") == third["chunks_created"],
          f"source count={api._indexed_chunk_count('notes.txt')} "
          f"vs chunks_created={third['chunks_created']}")
    check("collection total accounts for the swap exactly",
          stored_count() == before - old_src + third["chunks_created"],
          f"{before} - {old_src} + {third['chunks_created']} = {stored_count()}")
    check("other files untouched by the swap",
          api._indexed_chunk_count("outside.txt") > 0)

    print("\n=== 5. delete + download honour the same guard ===")
    leak = api.download_document("..\\..\\..\\Windows\\win.ini")
    check("download of traversal path is refused", isinstance(leak, dict) and leak.get("status") == "not_found",
          str(type(leak)))
    real = api.download_document("notes.txt")
    check("legitimate download still works", not isinstance(real, dict), str(real))

    doomed = tmp / "doomed.txt"
    doomed.write_text("must survive")
    api.delete_document("..\\..\\doomed.txt")
    check("delete of traversal path removed nothing outside", doomed.exists())
    api.delete_document("notes.txt")
    check("legitimate delete still works", not (api.DOCUMENTS_DIR / "notes.txt").exists())

finally:
    shutil.rmtree(tmp, ignore_errors=True)

print()
if FAILS:
    print(f"[FAIL] {len(FAILS)} check(s) failed: {FAILS}")
    sys.exit(1)
print("[OK] all upload safety checks passed")
