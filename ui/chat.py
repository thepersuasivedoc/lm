"""Chat UI: streaming responses, tool-call indicators, citations, handoff button."""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Generator

import streamlit as st

from agents import expert as expert_agent
from agents import storyteller as storyteller_agent
from config import HANDOFF_CONTEXT_TURNS
from hermes.memory import SessionDB
from storage import notebooks as notebooks_storage
from storage.notebooks import Message, Notebook


TOOL_LABELS = {
    "pubmed_search": "\U0001f9ec Querying PubMed",
    "file_search": "\U0001f50e Searching uploaded sources",
}


def _ensure_session_state() -> None:
    st.session_state.setdefault("agent_mode", "expert")
    st.session_state.setdefault("access_mode", "strict")
    st.session_state.setdefault("teach_mode", "explain")
    st.session_state.setdefault("script_format", "youtube_long")
    st.session_state.setdefault("expert_messages", [])
    st.session_state.setdefault("storyteller_messages", [])
    st.session_state.setdefault("handoff_packet", None)


@st.cache_resource
def _get_db() -> SessionDB:
    return SessionDB()


def _get_session_id(nb: Notebook, pane: str, first_message: str) -> str:
    """Lazily start a Hermes session for this (notebook, pane). Per-process scope."""
    key = f"hermes_sid_{nb.id}_{pane}"
    sid = st.session_state.get(key)
    if sid:
        return sid
    db = _get_db()
    sid = db.start_session(nb.id, pane=pane, title=first_message[:60].strip() or None)
    st.session_state[key] = sid
    return sid


def _render_citation(c: dict) -> str:
    t = c.get("type", "")
    if t == "file_search":
        title = c.get("title", "source")
        return f"\U0001f4d8 {title}"
    if t == "pubmed":
        title = c.get("title", c.get("pmid", "PubMed"))
        return f"\U0001f9ec {title}"
    return str(c)


def _render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Citations ({len(citations)})"):
        for c in citations:
            st.markdown("- " + _render_citation(c))


def _render_history(messages: list[Message]) -> None:
    for m in messages:
        ui_role = "user" if m.role == "user" else "assistant"
        with st.chat_message(ui_role):
            st.markdown(m.content)
            if m.citations:
                _render_citations(m.citations)


def _stream_agent(events: Generator[dict, None, None]) -> tuple[str, list[dict]]:
    """Consume an agent event generator and render to the current chat_message block.

    Returns (full_text, citations).
    """
    text_box = st.empty()
    tool_box = st.empty()
    full_text = ""
    citations: list[dict] = []
    active_tools: list[str] = []

    for event in events:
        kind = event.get("type")
        if kind == "text":
            full_text += event["content"]
            text_box.markdown(full_text + " ▌")
        elif kind == "tool_call":
            label = TOOL_LABELS.get(event.get("name", ""), f"\U0001f527 {event.get('name')}")
            active_tools.append(label)
            tool_box.info(" • ".join(active_tools))
        elif kind == "tool_result":
            # Drop the active indicator for that tool when it completes.
            label = TOOL_LABELS.get(event.get("name", ""), f"\U0001f527 {event.get('name')}")
            if label in active_tools:
                active_tools.remove(label)
            tool_box.info(" • ".join(active_tools) if active_tools else "")
        elif kind == "citations":
            citations.extend(event.get("citations", []))
        elif kind == "done":
            break

    text_box.markdown(full_text)
    tool_box.empty()
    return full_text, citations


def _build_handoff_packet(nb: Notebook, user_brief: str) -> str:
    recent = nb.expert_chat[-HANDOFF_CONTEXT_TURNS:]
    turns = [{"role": m.role, "content": m.content} for m in recent]
    sources = [asdict(s) for s in nb.sources]
    return storyteller_agent.build_handoff_packet(
        expert_chat_turns=turns,
        sources_digest=sources,
        user_brief=user_brief,
    )


def _handle_expert_turn(nb: Notebook, user_input: str) -> None:
    user_msg = Message(role="user", content=user_input, ts=time.time())
    notebooks_storage.append_message(nb, "expert", user_msg)
    st.session_state.expert_messages = nb.expert_chat[:]
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build messages payload (expert-only history, alternating user/expert).
    payload = [{"role": m.role, "content": m.content} for m in nb.expert_chat]

    with st.chat_message("assistant"):
        events = expert_agent.chat_with_expert(
            messages=payload,
            store_name=nb.file_search_store_name,
            access_mode=st.session_state.access_mode,
            teach_mode=st.session_state.teach_mode,
        )
        text, citations = _stream_agent(events)
        _render_citations(citations)

    assistant_msg = Message(role="expert", content=text, ts=time.time(), citations=citations)
    notebooks_storage.append_message(nb, "expert", assistant_msg)
    st.session_state.expert_messages = nb.expert_chat[:]

    db = _get_db()
    sid = _get_session_id(nb, "expert", user_input)
    db.append_turn(sid, "user", user_input)
    db.append_turn(sid, "expert", text)


def _handle_storyteller_turn(nb: Notebook, user_input: str) -> None:
    # Build/refresh handoff packet on every turn so the cache stays valid as the
    # expert chat evolves.
    handoff = _build_handoff_packet(nb, user_input)
    st.session_state.handoff_packet = handoff

    user_msg = Message(role="user", content=user_input, ts=time.time())
    notebooks_storage.append_message(nb, "storyteller", user_msg)
    with st.chat_message("user"):
        st.markdown(user_input)

    payload = [{"role": m.role, "content": m.content} for m in nb.storyteller_chat]

    with st.chat_message("assistant"):
        events = storyteller_agent.generate_script_ideas(
            messages=payload,
            handoff_packet=handoff,
            format_=st.session_state.script_format,
        )
        text, citations = _stream_agent(events)
        _render_citations(citations)

    assistant_msg = Message(role="storyteller", content=text, ts=time.time(), citations=citations)
    notebooks_storage.append_message(nb, "storyteller", assistant_msg)

    db = _get_db()
    sid = _get_session_id(nb, "storyteller", user_input)
    db.append_turn(sid, "user", user_input)
    db.append_turn(sid, "storyteller", text)


def _render_handoff_button(nb: Notebook) -> None:
    if st.session_state.agent_mode != "expert":
        return
    if not nb.expert_chat:
        return
    if st.button("\U0001f3ac Send to Storyteller", help="Switch to the storyteller agent and seed it with this conversation"):
        # Defer the mode switch: the radio widget in the sidebar has already
        # instantiated st.session_state.agent_mode this run, so we can't write
        # to it directly. Sentinel is consumed at the top of render_sidebar
        # on the next run, before the widget is rebuilt.
        st.session_state._mode_switch_pending = "storyteller"
        st.toast("Switched to Storyteller. Tell it what kind of script you want.", icon="\U0001f3ac")
        st.rerun()


def render_chat(nb: Notebook) -> None:
    _ensure_session_state()

    mode = st.session_state.agent_mode
    if mode == "expert":
        st.subheader("Expert tutor")
        st.caption(f"Mode: **{st.session_state.access_mode}** · Style: **{st.session_state.teach_mode}**")
    else:
        fmt_labels = {
            "youtube_long": "YouTube long-form",
            "shorts": "Shorts (30-90s)",
            "podcast": "Podcast",
        }
        st.subheader("Storyteller")
        st.caption(f"Format: **{fmt_labels[st.session_state.script_format]}**")

    history = nb.expert_chat if mode == "expert" else nb.storyteller_chat
    _render_history(history)

    _render_handoff_button(nb)

    placeholder = (
        "Ask anything from your sources..."
        if mode == "expert"
        else "Brief the storyteller: what kind of script do you want?"
    )
    user_input = st.chat_input(placeholder)
    if not user_input:
        return

    if mode == "expert":
        _handle_expert_turn(nb, user_input)
    else:
        _handle_storyteller_turn(nb, user_input)
