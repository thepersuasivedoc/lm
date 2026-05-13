"""Gemini File Search wrapper — create stores, upload PDFs/text, query, delete."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from typing import Iterable

from google import genai
from google.genai import types


def _client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


@dataclass
class IndexedFile:
    name: str            # full resource name: fileSearchStores/.../documents/...
    display_name: str
    size_bytes: int
    state: str           # e.g. "ACTIVE", "PROCESSING", "FAILED"


def create_store(display_name: str) -> str:
    """Create a File Search store and return its resource name (use as store_id)."""
    client = _client()
    store = client.file_search_stores.create(
        config={"display_name": display_name},
    )
    return store.name


def delete_store(store_name: str) -> None:
    client = _client()
    client.file_search_stores.delete(name=store_name, config={"force": True})


def _find_doc(store_name: str, display_name: str) -> "IndexedFile | None":
    for doc in list_files(store_name):
        if doc.display_name == display_name:
            return doc
    return None


def _poll_until_terminal(
    client: genai.Client,
    operation,
    store_name: str,
    display_name: str,
    timeout_sec: int,
    poll_sec: int,
) -> str:
    """Poll Gemini until the doc reaches a terminal state.

    Prefers the document state (faster, more honest) over the operation's done
    flag — Gemini occasionally lags the operation marker behind the doc's real
    state, which hangs the synchronous caller. Returns the resource name when
    the doc reaches STATE_ACTIVE.
    """
    deadline = time.time() + timeout_sec
    while True:
        doc = _find_doc(store_name, display_name)
        if doc is not None:
            state = str(doc.state).upper()
            if "ACTIVE" in state:
                return doc.name
            if "FAILED" in state and getattr(operation, "done", False):
                err = getattr(operation, "error", None)
                raise RuntimeError(
                    f"Gemini indexing failed for {display_name}"
                    + (f": {err}" if err else "")
                )
        if time.time() > deadline:
            raise TimeoutError(
                f"File Search indexing did not finish in time ({timeout_sec}s) for {display_name}."
            )
        time.sleep(poll_sec)
        try:
            operation = client.operations.get(operation)
        except Exception:
            # Operation tracking is best-effort; the doc state is authoritative.
            pass


def upload_pdf(
    store_name: str,
    pdf_bytes: bytes,
    display_name: str,
    timeout_sec: int = 180,
    poll_sec: int = 2,
) -> str:
    """Upload a PDF to the store, block until indexed, return the document resource name.

    Dedup-aware: if an ACTIVE document with the same display_name already exists,
    skip the upload and return the existing resource. This makes partial-upload
    retries idempotent.
    """
    existing = _find_doc(store_name, display_name)
    if existing is not None and "ACTIVE" in str(existing.state).upper():
        return existing.name

    client = _client()
    bio = io.BytesIO(pdf_bytes)
    bio.name = display_name if display_name.lower().endswith(".pdf") else f"{display_name}.pdf"

    operation = client.file_search_stores.upload_to_file_search_store(
        file=bio,
        file_search_store_name=store_name,
        config={"display_name": display_name, "mime_type": "application/pdf"},
    )
    return _poll_until_terminal(client, operation, store_name, display_name, timeout_sec, poll_sec)


def upload_text(
    store_name: str,
    text: str,
    display_name: str,
    timeout_sec: int = 180,
    poll_sec: int = 2,
) -> str:
    """Upload plain text (e.g., a YouTube transcript) as a .txt document."""
    existing = _find_doc(store_name, display_name)
    if existing is not None and "ACTIVE" in str(existing.state).upper():
        return existing.name

    client = _client()
    bio = io.BytesIO(text.encode("utf-8"))
    bio.name = display_name if display_name.lower().endswith(".txt") else f"{display_name}.txt"

    operation = client.file_search_stores.upload_to_file_search_store(
        file=bio,
        file_search_store_name=store_name,
        config={"display_name": display_name, "mime_type": "text/plain"},
    )
    return _poll_until_terminal(client, operation, store_name, display_name, timeout_sec, poll_sec)


def list_files(store_name: str) -> list[IndexedFile]:
    client = _client()
    result: list[IndexedFile] = []
    page = client.file_search_stores.documents.list(parent=store_name)
    # google-genai pages: iterate items, handle pagination if present.
    for doc in _iter_pages(page):
        result.append(
            IndexedFile(
                name=doc.name,
                display_name=getattr(doc, "display_name", "") or "",
                size_bytes=int(getattr(doc, "size_bytes", 0) or 0),
                state=str(getattr(doc, "state", "") or ""),
            )
        )
    return result


def _iter_pages(pager) -> Iterable:
    """Iterate paginated results from the google-genai SDK uniformly."""
    if hasattr(pager, "__iter__"):
        yield from pager
        return
    # Fallback for pagers exposing .next() or similar.
    while pager:
        for item in getattr(pager, "items", []):
            yield item
        pager = getattr(pager, "next_page", None) and pager.next_page() or None


def delete_file(document_name: str) -> None:
    client = _client()
    client.file_search_stores.documents.delete(name=document_name)


def get_store_size_bytes(store_name: str) -> int:
    return sum(f.size_bytes for f in list_files(store_name))


def file_search_tool(store_name: str) -> types.Tool:
    """Build the Gemini Tool object that scopes a generate_content call to one store."""
    return types.Tool(file_search=types.FileSearch(file_search_store_names=[store_name]))
