"""Gemini File Search wrapper — create stores, upload PDFs/text, query, delete."""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
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


def _wait_for_operation(client: genai.Client, operation, timeout_sec: int, poll_sec: int):
    deadline = time.time() + timeout_sec
    while not operation.done:
        if time.time() > deadline:
            raise TimeoutError("File Search indexing did not finish in time.")
        time.sleep(poll_sec)
        operation = client.operations.get(operation)
    return operation


def upload_pdf(
    store_name: str,
    pdf_bytes: bytes,
    display_name: str,
    timeout_sec: int = 180,
    poll_sec: int = 2,
) -> str:
    """Upload a PDF to the store, block until indexed, return the document resource name."""
    client = _client()
    bio = io.BytesIO(pdf_bytes)
    bio.name = display_name if display_name.lower().endswith(".pdf") else f"{display_name}.pdf"

    operation = client.file_search_stores.upload_to_file_search_store(
        file=bio,
        file_search_store_name=store_name,
        config={"display_name": display_name, "mime_type": "application/pdf"},
    )
    operation = _wait_for_operation(client, operation, timeout_sec, poll_sec)

    # The completed operation's response carries the document resource.
    response = getattr(operation, "response", None)
    if response is not None and getattr(response, "name", None):
        return response.name
    # Fallback: list documents and return the most recently created with this display_name.
    for doc in list_files(store_name):
        if doc.display_name == display_name:
            return doc.name
    raise RuntimeError(f"Could not resolve document resource for {display_name}")


def upload_text(
    store_name: str,
    text: str,
    display_name: str,
    timeout_sec: int = 180,
    poll_sec: int = 2,
) -> str:
    """Upload plain text (e.g., a YouTube transcript) as a .txt document."""
    client = _client()
    bio = io.BytesIO(text.encode("utf-8"))
    bio.name = display_name if display_name.lower().endswith(".txt") else f"{display_name}.txt"

    operation = client.file_search_stores.upload_to_file_search_store(
        file=bio,
        file_search_store_name=store_name,
        config={"display_name": display_name, "mime_type": "text/plain"},
    )
    operation = _wait_for_operation(client, operation, timeout_sec, poll_sec)

    response = getattr(operation, "response", None)
    if response is not None and getattr(response, "name", None):
        return response.name
    for doc in list_files(store_name):
        if doc.display_name == display_name:
            return doc.name
    raise RuntimeError(f"Could not resolve document resource for {display_name}")


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
