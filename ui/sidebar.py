"""Sidebar UI: notebook switcher, source ingestion (Local / Drive / YouTube), mode toggles."""

from __future__ import annotations

import time

import streamlit as st

from config import FILE_SEARCH_FREE_TIER_GB
from ingestion import drive as drive_ingest
from ingestion import pdf as pdf_ingest
from ingestion import youtube as youtube_ingest
from storage import file_search
from storage import notebooks as notebooks_storage
from storage.notebooks import Notebook, Source


# ---------- Notebook switcher ----------

def _refresh_notebook(notebook_id: str) -> Notebook:
    return notebooks_storage.load_notebook(notebook_id)


def _render_notebook_picker() -> Notebook | None:
    all_nbs = notebooks_storage.list_notebooks()

    st.subheader("Notebooks")

    if not all_nbs:
        st.caption("No notebooks yet. Create one to start.")
    else:
        ids = [nb["id"] for nb in all_nbs]
        labels = [f"{nb['name']} ({nb['source_count']} sources)" for nb in all_nbs]
        current = st.session_state.get("current_notebook_id")
        idx = ids.index(current) if current in ids else 0
        choice = st.selectbox("Active notebook", options=range(len(ids)), index=idx, format_func=lambda i: labels[i])
        st.session_state.current_notebook_id = ids[choice]

    col_new, col_del = st.columns(2)
    with col_new:
        with st.popover("➕ New", use_container_width=True):
            new_name = st.text_input("Notebook name", key="new_nb_name", placeholder="e.g. Cardiology Boards")
            if st.button("Create", key="create_nb_btn", use_container_width=True, type="primary"):
                if new_name.strip():
                    nb = notebooks_storage.create_notebook(new_name.strip())
                    st.session_state.current_notebook_id = nb.id
                    st.session_state.expert_messages = []
                    st.session_state.storyteller_messages = []
                    st.rerun()

    with col_del:
        if st.session_state.get("current_notebook_id"):
            with st.popover("\U0001f5d1 Delete", use_container_width=True):
                st.warning("This deletes the notebook AND its File Search store.")
                if st.button("Confirm delete", key="del_nb_btn", use_container_width=True, type="primary"):
                    notebooks_storage.delete_notebook(st.session_state.current_notebook_id)
                    st.session_state.current_notebook_id = None
                    st.session_state.expert_messages = []
                    st.session_state.storyteller_messages = []
                    st.rerun()

    if st.session_state.get("current_notebook_id"):
        return _refresh_notebook(st.session_state.current_notebook_id)
    return None


# ---------- Storage usage ----------

def _render_storage_usage(nb: Notebook) -> None:
    try:
        bytes_used = file_search.get_store_size_bytes(nb.file_search_store_name)
    except Exception:
        st.caption("Storage usage unavailable.")
        return
    total_bytes = FILE_SEARCH_FREE_TIER_GB * 1024 ** 3
    pct = min(1.0, bytes_used / total_bytes) if total_bytes else 0.0
    mb_used = bytes_used / (1024 ** 2)
    st.progress(pct, text=f"Storage: {mb_used:,.1f} MB / {FILE_SEARCH_FREE_TIER_GB} GB")
    if pct > 0.8:
        st.warning("Approaching free-tier limit. Remove old sources or upgrade your Gemini tier.")


# ---------- Source ingestion ----------

def _render_local_pdf_uploader(nb: Notebook) -> None:
    uploaded = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{nb.id}",
        label_visibility="collapsed",
    )
    if uploaded and st.button("Add to notebook", key=f"pdf_add_{nb.id}", use_container_width=True, type="primary"):
        files = [(f.name, f.read()) for f in uploaded]
        with st.status(f"Indexing {len(files)} PDF(s)...", expanded=True) as status:
            for filename, data in files:
                status.write(f"Uploading {filename}...")
                res = pdf_ingest.ingest_pdf(nb.file_search_store_name, data, filename)
                if res.status == "ready":
                    status.write(f"✅ {filename} indexed")
                    notebooks_storage.add_source(nb, Source(
                        document_name=res.document_name,
                        display_name=filename,
                        origin="local",
                        added_at=time.time(),
                    ))
                else:
                    status.write(f"❌ {filename}: {res.error}")
            status.update(label="Done", state="complete")
        st.rerun()


def _render_drive_picker(nb: Notebook) -> None:
    if not drive_ingest.is_connected():
        st.caption("Connect your Google Drive to import textbook PDFs.")
        if st.button("Connect Google Drive", key="drive_connect", use_container_width=True):
            try:
                with st.spinner("Opening browser for Google sign-in..."):
                    drive_ingest.connect()
                st.success("Connected.")
                st.rerun()
            except Exception as e:
                st.error(f"Connection failed: {e}")
        return

    if st.button("Disconnect Drive", key="drive_disconnect"):
        drive_ingest.disconnect()
        st.rerun()

    # Folder browser state.
    breadcrumbs = st.session_state.setdefault("drive_breadcrumbs", [("root", "My Drive")])
    current_id, current_name = breadcrumbs[-1]

    crumb_strs = " / ".join(name for _, name in breadcrumbs)
    st.caption(f"Folder: {crumb_strs}")

    if len(breadcrumbs) > 1 and st.button("⬅ Up", key="drive_up"):
        breadcrumbs.pop()
        st.rerun()

    try:
        folders = drive_ingest.list_folders(current_id)
        pdfs = drive_ingest.list_pdfs(current_id)
    except Exception as e:
        st.error(f"Drive list failed: {e}")
        return

    if folders:
        st.caption("Folders")
        for f in folders:
            if st.button(f"\U0001f4c1 {f.name}", key=f"drive_folder_{f.id}", use_container_width=True):
                breadcrumbs.append((f.id, f.name))
                st.rerun()

    if not pdfs:
        st.caption("No PDFs in this folder.")
        return

    st.caption("PDFs in this folder")
    pdf_labels = [f"{p.name} ({p.size / 1024 / 1024:.1f} MB)" for p in pdfs]
    selected_idxs = st.multiselect(
        "Select PDFs to import",
        options=range(len(pdfs)),
        format_func=lambda i: pdf_labels[i],
        key=f"drive_select_{current_id}",
        label_visibility="collapsed",
    )
    if selected_idxs and st.button(f"Import {len(selected_idxs)} file(s)", key=f"drive_import_{current_id}",
                                    use_container_width=True, type="primary"):
        chosen = [pdfs[i] for i in selected_idxs]
        with st.status(f"Importing {len(chosen)} PDF(s) from Drive...", expanded=True) as status:
            for f in chosen:
                status.write(f"Downloading {f.name}...")
                try:
                    data = drive_ingest.download_pdf(f.id)
                except Exception as e:
                    status.write(f"❌ {f.name}: download failed: {e}")
                    continue
                status.write(f"Indexing {f.name}...")
                res = pdf_ingest.ingest_pdf(nb.file_search_store_name, data, f.name)
                if res.status == "ready":
                    status.write(f"✅ {f.name} indexed")
                    notebooks_storage.add_source(nb, Source(
                        document_name=res.document_name,
                        display_name=f.name,
                        origin="drive",
                        added_at=time.time(),
                        extra={"drive_file_id": f.id},
                    ))
                else:
                    status.write(f"❌ {f.name}: {res.error}")
            status.update(label="Done", state="complete")
        st.rerun()


def _render_youtube_input(nb: Notebook) -> None:
    url = st.text_input("YouTube URL", key=f"yt_url_{nb.id}", placeholder="https://youtube.com/watch?v=...")
    if url and st.button("Add lecture", key=f"yt_add_{nb.id}", use_container_width=True, type="primary"):
        with st.status(f"Fetching transcript from {url}...", expanded=True) as status:
            res = youtube_ingest.ingest_youtube(nb.file_search_store_name, url)
            if res.status == "ready":
                status.write(f"✅ Indexed: {res.title}")
                notebooks_storage.add_source(nb, Source(
                    document_name=res.document_name,
                    display_name=res.title,
                    origin="youtube",
                    added_at=time.time(),
                    extra={"url": res.url, "video_id": res.video_id},
                ))
                status.update(label="Done", state="complete")
                st.rerun()
            else:
                status.write(f"❌ {res.error}")
                status.update(label="Failed", state="error")


def _render_source_list(nb: Notebook) -> None:
    if not nb.sources:
        st.caption("No sources yet. Add some above.")
        return
    origin_icons = {"local": "\U0001f4c1", "drive": "\U0001f7e2", "youtube": "\U0001f393"}
    for src in nb.sources:
        col_label, col_remove = st.columns([5, 1])
        icon = origin_icons.get(src.origin, "\U0001f4c4")
        col_label.markdown(f"{icon} {src.display_name}")
        if col_remove.button("✖", key=f"rm_{src.document_name}", help="Remove"):
            notebooks_storage.remove_source(nb, src.document_name)
            st.rerun()


# ---------- Mode toggles ----------

def _render_mode_toggles() -> None:
    st.subheader("Mode")
    agent_mode = st.radio(
        "Agent",
        options=["expert", "storyteller"],
        format_func=lambda x: "Expert (tutor)" if x == "expert" else "Storyteller (scripts)",
        key="agent_mode",
        horizontal=True,
    )

    if agent_mode == "expert":
        st.radio(
            "Access mode",
            options=["strict", "extended"],
            format_func=lambda x: "Strict (your sources only)" if x == "strict" else "Extended (+ PubMed, Web)",
            key="access_mode",
        )
        st.radio(
            "Teaching style",
            options=["explain", "summarize", "teach"],
            format_func=lambda x: x.capitalize(),
            key="teach_mode",
            horizontal=True,
        )
    else:
        st.radio(
            "Script format",
            options=["youtube_long", "shorts", "podcast"],
            format_func=lambda x: {
                "youtube_long": "YouTube long-form",
                "shorts": "Shorts (30-90s)",
                "podcast": "Podcast",
            }[x],
            key="script_format",
        )


# ---------- Public entry ----------

def render_sidebar() -> Notebook | None:
    with st.sidebar:
        nb = _render_notebook_picker()

        if nb is None:
            return None

        st.divider()
        _render_storage_usage(nb)

        st.divider()
        st.subheader("Sources")
        tab_local, tab_drive, tab_yt = st.tabs(["Local", "Drive", "YouTube"])
        with tab_local:
            _render_local_pdf_uploader(nb)
        with tab_drive:
            _render_drive_picker(nb)
        with tab_yt:
            _render_youtube_input(nb)

        st.markdown("**In this notebook:**")
        _render_source_list(nb)

        st.divider()
        _render_mode_toggles()

    return nb
