"""Google Drive ingestion: OAuth + browse + download PDFs into a notebook's File Search store."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Iterable

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from config import DRIVE_CREDENTIALS_PATH, DRIVE_SCOPES, DRIVE_TOKEN_PATH
from ingestion import pdf as pdf_ingest
from ingestion.pdf import IngestResult


@dataclass
class DriveFile:
    id: str
    name: str
    size: int
    modified_time: str
    mime_type: str


@dataclass
class DriveFolder:
    id: str
    name: str


def _load_credentials() -> Credentials | None:
    if DRIVE_TOKEN_PATH.exists():
        try:
            return Credentials.from_authorized_user_file(str(DRIVE_TOKEN_PATH), DRIVE_SCOPES)
        except Exception:
            return None
    return None


def _save_credentials(creds: Credentials) -> None:
    DRIVE_TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRIVE_TOKEN_PATH.write_text(creds.to_json())


def is_connected() -> bool:
    creds = _load_credentials()
    if creds is None:
        return False
    return bool(creds.valid or creds.refresh_token)


def connect() -> None:
    """Run the OAuth flow if needed. Opens a browser tab for consent."""
    if not DRIVE_CREDENTIALS_PATH.exists():
        raise RuntimeError(
            f"Drive client secrets not found at {DRIVE_CREDENTIALS_PATH}. "
            "Create an OAuth 2.0 Client ID (Desktop app) in Google Cloud Console, "
            "download credentials.json, and place it at that path."
        )
    creds = _load_credentials()
    if creds and creds.valid:
        return
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_credentials(creds)
        return
    flow = InstalledAppFlow.from_client_secrets_file(str(DRIVE_CREDENTIALS_PATH), DRIVE_SCOPES)
    creds = flow.run_local_server(port=0)
    _save_credentials(creds)


def disconnect() -> None:
    if DRIVE_TOKEN_PATH.exists():
        DRIVE_TOKEN_PATH.unlink()


def _service():
    creds = _load_credentials()
    if not creds:
        raise RuntimeError("Drive is not connected. Call connect() first.")
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        _save_credentials(creds)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_folders(parent_id: str = "root") -> list[DriveFolder]:
    svc = _service()
    q = (
        f"'{parent_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and "
        "trashed = false"
    )
    folders: list[DriveFolder] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=q,
            fields="nextPageToken, files(id, name)",
            pageSize=100,
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            folders.append(DriveFolder(id=f["id"], name=f["name"]))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return sorted(folders, key=lambda x: x.name.lower())


def list_pdfs(folder_id: str = "root") -> list[DriveFile]:
    svc = _service()
    q = (
        f"'{folder_id}' in parents and "
        "(mimeType = 'application/pdf' or fileExtension = 'pdf') and "
        "trashed = false"
    )
    files: list[DriveFile] = []
    page_token = None
    while True:
        resp = svc.files().list(
            q=q,
            fields="nextPageToken, files(id, name, size, modifiedTime, mimeType)",
            pageSize=100,
            pageToken=page_token,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
        ).execute()
        for f in resp.get("files", []):
            files.append(
                DriveFile(
                    id=f["id"],
                    name=f["name"],
                    size=int(f.get("size", 0) or 0),
                    modified_time=f.get("modifiedTime", ""),
                    mime_type=f.get("mimeType", ""),
                )
            )
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return sorted(files, key=lambda x: x.name.lower())


def download_pdf(file_id: str) -> bytes:
    svc = _service()
    request = svc.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def import_drive_pdfs(
    store_name: str,
    files: Iterable[DriveFile],
    progress_cb: Callable[[str, str], None] | None = None,
) -> list[IngestResult]:
    results: list[IngestResult] = []
    for f in files:
        if progress_cb:
            progress_cb(f.name, "downloading")
        try:
            data = download_pdf(f.id)
        except Exception as e:
            results.append(IngestResult(filename=f.name, document_name=None, status="failed", error=str(e)))
            if progress_cb:
                progress_cb(f.name, "failed")
            continue
        if progress_cb:
            progress_cb(f.name, "uploading")
        res = pdf_ingest.ingest_pdf(store_name, data, f.name)
        results.append(res)
        if progress_cb:
            progress_cb(f.name, res.status)
    return results
