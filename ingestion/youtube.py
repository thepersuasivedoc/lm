"""YouTube ingestion: fetch transcript + metadata, upload as timestamped .txt to File Search."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from config import INDEXING_POLL_INTERVAL_SEC, INDEXING_TIMEOUT_SEC
from storage import file_search


@dataclass
class YouTubeIngestResult:
    video_id: str
    title: str
    url: str
    document_name: str | None
    status: str            # "ready" | "failed"
    error: str | None = None


def parse_video_id(url: str) -> str | None:
    """Extract the 11-char video id from a YouTube URL (watch, youtu.be, shorts)."""
    if not url:
        return None
    p = urlparse(url.strip())
    host = (p.hostname or "").lower()
    if host in ("youtu.be",):
        vid = p.path.lstrip("/").split("/")[0]
        return vid if re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or "") else None
    if "youtube.com" in host:
        qs = parse_qs(p.query)
        if "v" in qs and re.fullmatch(r"[A-Za-z0-9_-]{11}", qs["v"][0]):
            return qs["v"][0]
        # /shorts/<id> or /embed/<id>
        parts = [seg for seg in p.path.split("/") if seg]
        for i, seg in enumerate(parts):
            if seg in ("shorts", "embed", "v") and i + 1 < len(parts):
                candidate = parts[i + 1]
                if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
                    return candidate
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    return None


def _fetch_metadata(url: str) -> dict:
    opts = {"skip_download": True, "quiet": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title") or "Untitled",
        "channel": info.get("uploader") or info.get("channel") or "",
        "duration": _format_duration(info.get("duration") or 0),
    }


def _format_duration(seconds: int | float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _format_timestamp(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _fetch_transcript(video_id: str) -> list[dict]:
    transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
    try:
        t = transcripts.find_manually_created_transcript(["en"])
    except Exception:
        t = transcripts.find_transcript(["en"])
    return t.fetch()


def _build_transcript_text(meta: dict, url: str, segments: list[dict]) -> str:
    header = (
        f"Title: {meta['title']}\n"
        f"Channel: {meta['channel']}\n"
        f"URL: {url}\n"
        f"Duration: {meta['duration']}\n\n"
        f"--- TRANSCRIPT ---\n"
    )
    lines = []
    for seg in segments:
        ts = _format_timestamp(seg.get("start", 0.0))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        if text:
            lines.append(f"[{ts}] {text}")
    return header + "\n".join(lines)


def ingest_youtube(store_name: str, url: str) -> YouTubeIngestResult:
    video_id = parse_video_id(url)
    if not video_id:
        return YouTubeIngestResult(
            video_id="", title="", url=url, document_name=None,
            status="failed", error="Could not parse a YouTube video ID from that URL.",
        )

    try:
        meta = _fetch_metadata(url)
    except Exception as e:
        return YouTubeIngestResult(
            video_id=video_id, title="", url=url, document_name=None,
            status="failed", error=f"Failed to fetch video metadata: {e}",
        )

    try:
        segments = _fetch_transcript(video_id)
    except TranscriptsDisabled:
        return YouTubeIngestResult(
            video_id=video_id, title=meta["title"], url=url, document_name=None,
            status="failed", error="This video has transcripts disabled.",
        )
    except NoTranscriptFound:
        return YouTubeIngestResult(
            video_id=video_id, title=meta["title"], url=url, document_name=None,
            status="failed", error="No transcript available for this video.",
        )
    except VideoUnavailable:
        return YouTubeIngestResult(
            video_id=video_id, title=meta["title"], url=url, document_name=None,
            status="failed", error="Video is unavailable (private, deleted, or region-locked).",
        )
    except Exception as e:
        return YouTubeIngestResult(
            video_id=video_id, title=meta["title"], url=url, document_name=None,
            status="failed", error=f"Transcript fetch failed: {e}",
        )

    text = _build_transcript_text(meta, url, segments)
    display_name = f"{meta['title']} [{video_id}]"

    try:
        doc_name = file_search.upload_text(
            store_name=store_name,
            text=text,
            display_name=display_name,
            timeout_sec=INDEXING_TIMEOUT_SEC,
            poll_sec=INDEXING_POLL_INTERVAL_SEC,
        )
    except Exception as e:
        return YouTubeIngestResult(
            video_id=video_id, title=meta["title"], url=url, document_name=None,
            status="failed", error=f"Upload to File Search failed: {e}",
        )

    return YouTubeIngestResult(
        video_id=video_id,
        title=meta["title"],
        url=url,
        document_name=doc_name,
        status="ready",
    )
