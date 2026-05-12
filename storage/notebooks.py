"""Notebook metadata persistence: each notebook = one folder, one File Search store."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Literal

from config import NOTEBOOKS_DIR
from storage import file_search


SourceOrigin = Literal["local", "drive", "youtube"]


@dataclass
class Source:
    document_name: str       # File Search resource name (fileSearchStores/.../documents/...)
    display_name: str
    origin: SourceOrigin
    added_at: float
    extra: dict = field(default_factory=dict)   # e.g. {"url": "..."} for YouTube


@dataclass
class Message:
    role: str                # "user" | "expert" | "storyteller"
    content: str
    ts: float
    citations: list[dict] = field(default_factory=list)
    tool_calls: list[dict] = field(default_factory=list)


@dataclass
class Notebook:
    id: str
    name: str
    file_search_store_name: str
    created_at: float
    updated_at: float
    sources: list[Source] = field(default_factory=list)
    expert_chat: list[Message] = field(default_factory=list)
    storyteller_chat: list[Message] = field(default_factory=list)


def _meta_path(notebook_id: str):
    return NOTEBOOKS_DIR / notebook_id / "meta.json"


def _ensure_dir(notebook_id: str) -> None:
    (NOTEBOOKS_DIR / notebook_id).mkdir(parents=True, exist_ok=True)


def create_notebook(name: str) -> Notebook:
    notebook_id = uuid.uuid4().hex[:12]
    store_name = file_search.create_store(display_name=f"lm-{notebook_id}-{name}"[:100])
    now = time.time()
    nb = Notebook(
        id=notebook_id,
        name=name,
        file_search_store_name=store_name,
        created_at=now,
        updated_at=now,
    )
    _ensure_dir(notebook_id)
    save_notebook(nb)
    return nb


def list_notebooks() -> list[dict]:
    if not NOTEBOOKS_DIR.exists():
        return []
    out: list[dict] = []
    for p in NOTEBOOKS_DIR.iterdir():
        if not p.is_dir():
            continue
        meta = p / "meta.json"
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            out.append({
                "id": data["id"],
                "name": data["name"],
                "source_count": len(data.get("sources", [])),
                "updated_at": data.get("updated_at", 0),
            })
        except Exception:
            continue
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return out


def load_notebook(notebook_id: str) -> Notebook:
    data = json.loads(_meta_path(notebook_id).read_text(encoding="utf-8"))
    return Notebook(
        id=data["id"],
        name=data["name"],
        file_search_store_name=data["file_search_store_name"],
        created_at=data.get("created_at", 0.0),
        updated_at=data.get("updated_at", 0.0),
        sources=[Source(**s) for s in data.get("sources", [])],
        expert_chat=[Message(**m) for m in data.get("expert_chat", [])],
        storyteller_chat=[Message(**m) for m in data.get("storyteller_chat", [])],
    )


def save_notebook(nb: Notebook) -> None:
    nb.updated_at = time.time()
    _ensure_dir(nb.id)
    _meta_path(nb.id).write_text(
        json.dumps(asdict(nb), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def add_source(nb: Notebook, source: Source) -> Notebook:
    nb.sources.append(source)
    save_notebook(nb)
    return nb


def remove_source(nb: Notebook, document_name: str) -> Notebook:
    src = next((s for s in nb.sources if s.document_name == document_name), None)
    if src is None:
        return nb
    try:
        file_search.delete_file(document_name)
    except Exception:
        pass
    nb.sources = [s for s in nb.sources if s.document_name != document_name]
    save_notebook(nb)
    return nb


def delete_notebook(notebook_id: str) -> None:
    nb = load_notebook(notebook_id)
    try:
        file_search.delete_store(nb.file_search_store_name)
    except Exception:
        pass
    # Remove the notebook folder (meta.json + anything else).
    import shutil
    shutil.rmtree(NOTEBOOKS_DIR / notebook_id, ignore_errors=True)


def append_message(nb: Notebook, agent: Literal["expert", "storyteller"], msg: Message) -> Notebook:
    if agent == "expert":
        nb.expert_chat.append(msg)
    else:
        nb.storyteller_chat.append(msg)
    save_notebook(nb)
    return nb
