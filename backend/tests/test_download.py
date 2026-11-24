"""Test document download functionality.

Run:  python test_download.py

Tests:
  1. Download endpoint serves files correctly
  2. Download tool generates correct URLs
  3. Missing files return appropriate errors
"""

from pathlib import Path
import sys
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


API_BASE = "http://127.0.0.1:8000"


def test_download_endpoint():
    print("\n=== Test 1: Download endpoint ===")

    # First, list documents to find one to download
    response = requests.get(f"{API_BASE}/documents")
    docs = response.json()

    if docs["count"] == 0:
        print("  No documents available to test download")
        print("[SKIP] No documents to download")
        return

    filename = docs["documents"][0]["filename"]
    print(f"  Testing download of: {filename}")

    # URL-encode the filename (handles spaces and special chars)
    from urllib.parse import quote
    encoded_filename = quote(filename, safe='')

    # Try to download
    response = requests.get(f"{API_BASE}/documents/{encoded_filename}/download")

    if response.status_code == 200:
        print(f"  Download successful: {len(response.content)} bytes")
        print("[OK] Download endpoint works")
    else:
        print(f"  Download failed: {response.status_code}")
        print(f"  Response: {response.text[:200]}")
        print("[FAIL] Download endpoint error")
        sys.exit(1)


def test_download_missing_file():
    print("\n=== Test 2: Download missing file ===")

    response = requests.get(f"{API_BASE}/documents/nonexistent.pdf/download")

    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "not_found":
            print("[OK] Missing file returns not_found")
        else:
            print(f"[FAIL] Unexpected response: {data}")
            sys.exit(1)
    else:
        print(f"[FAIL] Unexpected status: {response.status_code}")
        sys.exit(1)


def test_download_tool():
    print("\n=== Test 3: Download link tool ===")

    # Import the tool directly
    from agent_graph import get_document_download_link

    # Test with a real file
    from pathlib import Path
    documents_dir = Path(__file__).resolve().parent.parent / "data" / "documents"

    if documents_dir.exists():
        files = list(documents_dir.iterdir())
        if files:
            filename = files[0].name
            result = get_document_download_link.invoke({"filename": filename})
            print(f"  Tool result: {result}")

            if "http://127.0.0.1:8000/documents/" in result and filename in result:
                print("[OK] Tool generates correct download link")
            else:
                print("[FAIL] Tool output missing expected URL")
                sys.exit(1)
        else:
            print("  No files to test")
            print("[SKIP] No documents available")
    else:
        print("  Documents directory doesn't exist")
        print("[SKIP] No documents directory")


def test_download_tool_missing_file():
    print("\n=== Test 4: Download tool with missing file ===")

    from agent_graph import get_document_download_link

    result = get_document_download_link.invoke({"filename": "nonexistent.pdf"})
    print(f"  Tool result: {result}")

    if "not found" in result.lower():
        print("[OK] Tool handles missing file correctly")
    else:
        print("[FAIL] Tool should report file not found")
        sys.exit(1)


if __name__ == "__main__":
    print("Testing document download functionality...")

    try:
        # Check if API is running
        try:
            response = requests.get(f"{API_BASE}/health", timeout=2)
            if response.status_code != 200:
                print(f"\n[FAIL] API not running or unhealthy: {response.status_code}")
                print("Start the API with: python api.py")
                sys.exit(1)
        except requests.exceptions.ConnectionError:
            print(f"\n[FAIL] Cannot connect to API at {API_BASE}")
            print("Start the API with: python api.py")
            sys.exit(1)

        test_download_endpoint()
        test_download_missing_file()
        test_download_tool()
        test_download_tool_missing_file()

        print("\n" + "=" * 50)
        print("[OK] All download tests passed!")
        print("=" * 50)
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
