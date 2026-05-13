"""PDF ingestion → Gemini File Search, with auto-splitting for PDFs over Gemini's 100 MB per-file cap."""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

from pypdf import PdfReader, PdfWriter

from config import (
    FILE_SEARCH_MAX_FILE_MB,
    INDEXING_TIMEOUT_SEC,
    INDEXING_POLL_INTERVAL_SEC,
    PDF_SPLIT_TARGET_MB,
)
from storage import file_search


PDF_MAGIC = b"%PDF-"


@dataclass
class IngestResult:
    filename: str        # display name actually used on Gemini (includes "(part N of M)" suffix when split)
    document_name: str | None
    status: str          # "ready" | "failed"
    error: str | None = None


def _validate_magic(data: bytes, filename: str) -> str | None:
    if not data.startswith(PDF_MAGIC):
        return f"{filename}: not a valid PDF (missing magic bytes)"
    return None


def _split_pdf_if_oversize(pdf_bytes: bytes, filename: str) -> list[tuple[str, bytes]]:
    """Split a PDF into chunks ≤ PDF_SPLIT_TARGET_MB. Returns (display_name, bytes) list.

    For PDFs at or below the target size, returns a single-element list with the
    original filename unchanged. Larger PDFs split by page count, naming each
    chunk "<base> (part N of M, pp.X-Y).pdf".
    """
    target_bytes = PDF_SPLIT_TARGET_MB * 1024 * 1024
    if len(pdf_bytes) <= target_bytes:
        return [(filename, pdf_bytes)]

    reader = PdfReader(io.BytesIO(pdf_bytes))
    total_pages = len(reader.pages)
    if total_pages == 0:
        raise ValueError("PDF has zero pages")

    n_chunks = math.ceil(len(pdf_bytes) / target_bytes)
    pages_per_chunk = math.ceil(total_pages / n_chunks)

    base = filename[:-4] if filename.lower().endswith(".pdf") else filename

    chunks: list[tuple[str, bytes]] = []
    for i in range(n_chunks):
        start = i * pages_per_chunk
        end = min(start + pages_per_chunk, total_pages)
        if start >= total_pages:
            break

        writer = PdfWriter()
        for p in range(start, end):
            writer.add_page(reader.pages[p])
        buf = io.BytesIO()
        writer.write(buf)
        chunk_bytes = buf.getvalue()
        chunk_name = f"{base} (part {i + 1} of {n_chunks}, pp.{start + 1}-{end}).pdf"
        chunks.append((chunk_name, chunk_bytes))

    return chunks


def ingest_pdf(store_name: str, pdf_bytes: bytes, filename: str) -> list[IngestResult]:
    """Upload a PDF to a File Search store, auto-splitting if it exceeds Gemini's 100 MB cap.

    Returns one IngestResult per uploaded chunk. A small PDF yields a list of length 1.
    """
    err = _validate_magic(pdf_bytes, filename)
    if err:
        return [IngestResult(filename=filename, document_name=None, status="failed", error=err)]

    try:
        chunks = _split_pdf_if_oversize(pdf_bytes, filename)
    except Exception as e:
        return [IngestResult(
            filename=filename, document_name=None, status="failed",
            error=f"PDF split failed: {e}",
        )]

    hard_cap = FILE_SEARCH_MAX_FILE_MB * 1024 * 1024
    results: list[IngestResult] = []
    for chunk_name, chunk_bytes in chunks:
        if len(chunk_bytes) > hard_cap:
            results.append(IngestResult(
                filename=chunk_name, document_name=None, status="failed",
                error=f"chunk still exceeds {FILE_SEARCH_MAX_FILE_MB} MB after split ({len(chunk_bytes)/1024/1024:.1f} MB)",
            ))
            continue
        try:
            doc_name = file_search.upload_pdf(
                store_name=store_name,
                pdf_bytes=chunk_bytes,
                display_name=chunk_name,
                timeout_sec=INDEXING_TIMEOUT_SEC,
                poll_sec=INDEXING_POLL_INTERVAL_SEC,
            )
            results.append(IngestResult(filename=chunk_name, document_name=doc_name, status="ready"))
        except Exception as e:
            results.append(IngestResult(
                filename=chunk_name, document_name=None, status="failed", error=str(e),
            ))

    return results
