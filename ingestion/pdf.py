"""PDF ingestion → Gemini File Search."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from config import FILE_SEARCH_MAX_FILE_MB, INDEXING_TIMEOUT_SEC, INDEXING_POLL_INTERVAL_SEC
from storage import file_search


PDF_MAGIC = b"%PDF-"


@dataclass
class IngestResult:
    filename: str
    document_name: str | None
    status: str          # "ready" | "failed"
    error: str | None = None


def _validate_pdf(data: bytes, filename: str) -> str | None:
    if not data.startswith(PDF_MAGIC):
        return f"{filename}: not a valid PDF (missing magic bytes)"
    max_bytes = FILE_SEARCH_MAX_FILE_MB * 1024 * 1024
    if len(data) > max_bytes:
        return f"{filename}: {len(data) / 1024 / 1024:.1f} MB exceeds {FILE_SEARCH_MAX_FILE_MB} MB limit"
    return None


def ingest_pdf(store_name: str, pdf_bytes: bytes, filename: str) -> IngestResult:
    err = _validate_pdf(pdf_bytes, filename)
    if err:
        return IngestResult(filename=filename, document_name=None, status="failed", error=err)
    try:
        doc_name = file_search.upload_pdf(
            store_name=store_name,
            pdf_bytes=pdf_bytes,
            display_name=filename,
            timeout_sec=INDEXING_TIMEOUT_SEC,
            poll_sec=INDEXING_POLL_INTERVAL_SEC,
        )
        return IngestResult(filename=filename, document_name=doc_name, status="ready")
    except Exception as e:
        return IngestResult(filename=filename, document_name=None, status="failed", error=str(e))


def bulk_ingest_pdfs(
    store_name: str,
    files: Iterable[tuple[str, bytes]],
    progress_cb: Callable[[str, str], None] | None = None,
) -> list[IngestResult]:
    """Upload many PDFs sequentially. `files` is iterable of (filename, bytes).

    progress_cb(filename, status) is called per file with status in
    {"uploading", "ready", "failed"}.
    """
    results: list[IngestResult] = []
    for filename, data in files:
        if progress_cb:
            progress_cb(filename, "uploading")
        res = ingest_pdf(store_name, data, filename)
        results.append(res)
        if progress_cb:
            progress_cb(filename, res.status)
    return results
