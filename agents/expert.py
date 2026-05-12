"""Expert agent — Gemini with File Search (strict) or File Search + PubMed + Google Search (extended)."""

from __future__ import annotations

import os
from typing import Generator, Iterable, Literal

from google import genai
from google.genai import types

from config import EXPERT_MODEL
from agents.tools import (
    dispatch_gemini_function_call,
    gemini_google_search_tool,
    gemini_pubmed_tool,
)
from storage import file_search


AccessMode = Literal["strict", "extended"]
TeachMode = Literal["explain", "summarize", "teach"]


SYSTEM_BASE = """You are an expert medical teacher grounded in the user's uploaded sources \
(medical textbooks and lecture transcripts). You are supporting a medical student who \
depends on accurate, citable answers.

ABSOLUTE RULES:
1. EVERY clinical claim cites a source. No citation = do not make the claim.
2. Cite format depends on source:
   - Uploaded PDF: [Source: filename.pdf, p.12]
   - Lecture: [Lecture: title, 04:32]
   - PubMed: [PubMed: PMID 12345678 — first-author et al., journal year]
   - Web: [Web: domain.com — page title]
3. Never invent drug doses, mechanisms, contraindications, or guidelines.
4. Flag conflicting sources explicitly: "Source A says X; PubMed says Y — they differ."
"""

SYSTEM_STRICT = """\
ACCESS MODE: STRICT.
You have ONLY file_search available. If a topic isn't covered by the uploaded sources, \
say: "This isn't covered in your current sources. Upload more or switch to extended mode \
to consult PubMed/web." Do NOT draw on outside medical knowledge.
"""

SYSTEM_EXTENDED = """\
ACCESS MODE: EXTENDED.
Tool priority:
1. ALWAYS try the user's uploaded sources first (file_search runs implicitly on every turn).
2. If the uploaded sources don't cover the topic, call pubmed_search for peer-reviewed \
clinical evidence, or use google_search for general / anatomical / definitional gaps.
3. State explicitly which source produced each part of the answer.
4. Prefer uploaded sources + PubMed over arbitrary web pages.
"""

TEACH_PROMPTS = {
    "explain": (
        "TEACHING MODE: EXPLAIN. Give a clear, structured explanation. Define terms. "
        "Use analogies sparingly and only when grounded in source examples. Cite each claim."
    ),
    "summarize": (
        "TEACHING MODE: SUMMARIZE. Produce a tight hierarchical summary. Lead with the bottom line. "
        "Bullet key facts with citations. Be ruthless about brevity."
    ),
    "teach": (
        "TEACHING MODE: SOCRATIC. Start by asking the student a diagnostic question to find what "
        "they already know. Identify gaps. Quiz. Do NOT lecture unless asked. One question at a time."
    ),
}


def _system_prompt(access_mode: AccessMode, teach_mode: TeachMode) -> str:
    parts = [SYSTEM_BASE]
    parts.append(SYSTEM_STRICT if access_mode == "strict" else SYSTEM_EXTENDED)
    parts.append(TEACH_PROMPTS[teach_mode])
    return "\n\n".join(parts)


def _client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    return genai.Client(api_key=api_key)


def _build_tools(store_name: str, access_mode: AccessMode) -> list[types.Tool]:
    tools: list[types.Tool] = [file_search.file_search_tool(store_name)]
    if access_mode == "extended":
        tools.append(gemini_google_search_tool())
        tools.append(gemini_pubmed_tool())
    return tools


def _messages_to_contents(messages: Iterable[dict]) -> list[types.Content]:
    """Convert simple {role, content} messages to Gemini Content objects.

    Roles: 'user' or 'expert' (which becomes Gemini's 'model').
    """
    out: list[types.Content] = []
    for m in messages:
        role = "model" if m["role"] == "expert" else "user"
        out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return out


def chat_with_expert(
    messages: list[dict],
    store_name: str,
    access_mode: AccessMode,
    teach_mode: TeachMode,
) -> Generator[dict, None, None]:
    """Stream events from the expert agent.

    Event shapes:
      {"type": "text", "content": str}
      {"type": "tool_call", "name": str, "args": dict}
      {"type": "tool_result", "name": str, "result": dict}
      {"type": "citations", "citations": list[dict]}
      {"type": "done"}
    """
    client = _client()
    tools = _build_tools(store_name, access_mode)
    system_instruction = _system_prompt(access_mode, teach_mode)
    contents = _messages_to_contents(messages)

    max_tool_rounds = 4
    for _ in range(max_tool_rounds):
        stream = client.models.generate_content_stream(
            model=EXPERT_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
            ),
        )

        function_calls: list[types.FunctionCall] = []
        assistant_parts: list[types.Part] = []
        grounding_metadata = None
        produced_text = False

        for chunk in stream:
            if not chunk.candidates:
                continue
            cand = chunk.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        produced_text = True
                        assistant_parts.append(types.Part(text=part.text))
                        yield {"type": "text", "content": part.text}
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name:
                        function_calls.append(fc)
                        assistant_parts.append(types.Part(function_call=fc))
                        args = dict(fc.args) if fc.args else {}
                        yield {"type": "tool_call", "name": fc.name, "args": args}
            gm = getattr(cand, "grounding_metadata", None)
            if gm is not None:
                grounding_metadata = gm

        # Persist the assistant turn into the conversation contents.
        if assistant_parts:
            contents.append(types.Content(role="model", parts=assistant_parts))

        if not function_calls:
            if grounding_metadata is not None:
                yield {"type": "citations", "citations": _parse_grounding(grounding_metadata)}
            yield {"type": "done"}
            return

        # Execute each function call and feed responses back as a user turn.
        response_parts: list[types.Part] = []
        for fc in function_calls:
            args = dict(fc.args) if fc.args else {}
            result = dispatch_gemini_function_call(fc.name, args)
            yield {"type": "tool_result", "name": fc.name, "result": result}
            response_parts.append(
                types.Part(function_response=types.FunctionResponse(name=fc.name, response=result))
            )
        contents.append(types.Content(role="user", parts=response_parts))

    yield {"type": "text", "content": "\n\n(Stopped: tool-call loop exceeded max rounds.)"}
    yield {"type": "done"}


def _parse_grounding(gm) -> list[dict]:
    """Convert Gemini grounding metadata into a flat citation list for the UI."""
    citations: list[dict] = []

    # Google Search grounding chunks.
    chunks = getattr(gm, "grounding_chunks", []) or []
    for c in chunks:
        web = getattr(c, "web", None)
        if web is not None:
            citations.append({
                "type": "web",
                "title": getattr(web, "title", "") or "",
                "uri": getattr(web, "uri", "") or "",
            })
        retrieved = getattr(c, "retrieved_context", None)
        if retrieved is not None:
            citations.append({
                "type": "file_search",
                "title": getattr(retrieved, "title", "") or "",
                "uri": getattr(retrieved, "uri", "") or "",
                "text": getattr(retrieved, "text", "") or "",
            })

    return citations
