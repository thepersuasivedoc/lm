"""Streamlit entry point — wires sidebar + chat into a single page."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from config import NOTEBOOKS_DIR
from ui.chat import render_chat
from ui.sidebar import render_sidebar


load_dotenv()
NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)


def _check_api_keys() -> list[str]:
    missing: list[str] = []
    if not os.environ.get("GOOGLE_API_KEY"):
        missing.append("GOOGLE_API_KEY (Gemini)")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY (Claude)")
    return missing


def main() -> None:
    st.set_page_config(
        page_title="Medical Tutor + Storyteller",
        page_icon="\U0001f9ec",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Medical Tutor + Storyteller")

    missing = _check_api_keys()
    if missing:
        st.error(
            "Missing API keys: " + ", ".join(missing) +
            ". Copy `.env.example` to `.env` and fill in your keys, then restart."
        )
        st.stop()

    nb = render_sidebar()

    if nb is None:
        st.info("Create a notebook in the sidebar to get started.")
        return

    render_chat(nb)


if __name__ == "__main__":
    main()
