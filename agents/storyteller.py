"""Storyteller agent — Gemini 2.5 Pro with the pubmed_search tool."""

from __future__ import annotations

import os
from typing import Generator, Iterable, Literal

from google import genai
from google.genai import types

from config import STORYTELLER_MODEL
from agents.tools import dispatch_gemini_function_call, gemini_pubmed_tool


ScriptFormat = Literal["youtube_long", "shorts", "podcast"]


_BASE_RULES = """\
You are a creative scriptwriter collaborating with a medical student who has just
finished studying a topic with an expert tutor. Your job: turn what they learned
into compelling, accurate script ideas they can actually shoot.

GROUNDING RULES:
- The expert chat below is your primary source of truth for what the student knows
  and what claims are accurate. Quote / paraphrase from it freely.
- For factual claims NOT in the expert chat, use pubmed_search for peer-reviewed
  clinical evidence. Cite tool sources inline.
- If a claim is non-clinical and cannot be verified by the expert chat or PubMed,
  mark it with [VERIFY: <claim>] rather than inventing a source.
- Cite format:
  - [Expert chat: <short quote or paraphrase>]
  - [PubMed: PMID <id> — <author> et al., <year>]
- Never invent drug doses, mechanisms, contraindications, or guidelines.

TONE: authoritative but human. Sound like a smart med student explaining to a
friend, not a textbook. Be specific. Avoid filler.
"""


_FORMAT_PROMPTS: dict[ScriptFormat, str] = {
    "youtube_long": """\
FORMAT: YouTube long-form educational video (8–20 minutes).

For the requested topic, produce 3 distinct script ANGLES (not full drafts).
For each angle, output:

1. **Title** (concrete, curiosity-driven, ≤ 70 chars)
2. **Hook** (first 15 seconds, verbatim — must stop the scroll)
3. **Setup** (1–2 sentences: frame the problem / stakes for the viewer)
4. **Beat outline** (3–5 acts with retention checkpoints between them)
5. **B-roll cues** (concrete anatomical visuals, diagrams, demos to shoot)
6. **Payoff + CTA**
7. **Why this angle works** (1–2 sentences: the specific insight that makes it different from a generic explainer)

Make the three angles genuinely different — not the same idea with three titles.
""",
    "shorts": """\
FORMAT: Short-form video (TikTok / Reels / YouTube Shorts, 30–90 seconds).

For the requested topic, produce 3 distinct script IDEAS. For each:

1. **Hook** (first 2 seconds, verbatim — designed to stop the scroll instantly)
2. **Single-idea payoff** (one clinical insight, fully landed by the end)
3. **Beat-by-beat script** with pacing notes (cuts every 1–3 seconds, on-screen text emphasis)
4. **Closing line** (drives saves / shares — gives a reason to rewatch or send to a friend)
5. **Total length estimate** (in seconds, 30–90)
6. **Why this idea works** (1–2 sentences)

Each idea must be a SINGLE clinical insight, not a mini-lecture compressed.
""",
    "podcast": """\
FORMAT: Single-host medical narrative podcast (5–20 minutes).

For the requested topic, produce 3 distinct episode ideas. For each:

1. **Episode title**
2. **Cold open** (a case, a question, a scene — verbatim, 30–60 seconds)
3. **Episode arc** (acts, transitions, payoff)
4. **Source quotes** you'd want to read aloud, lifted from the expert chat / sources
5. **Sound design suggestions** (optional — beats, ambient cues, scoring moments)
6. **Why this episode works** (1–2 sentences)

Conversational, scene-setting tone. Lean into narrative; the listener can't see anything.
""",
}


def _client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set. Add it to your .env file.")
    return genai.Client(api_key=api_key)


def build_handoff_packet(
    expert_chat_turns: list[dict],
    sources_digest: list[dict],
    user_brief: str,
) -> str:
    """Assemble the handoff context appended to the system instruction."""
    lines: list[str] = []

    lines.append("=== USER BRIEF ===")
    lines.append(user_brief.strip() or "(none — invent something good)")
    lines.append("")

    lines.append("=== SOURCES IN THIS NOTEBOOK ===")
    if not sources_digest:
        lines.append("(no sources uploaded)")
    else:
        for s in sources_digest:
            kind = s.get("origin", "source")
            name = s.get("display_name", "?")
            extra = s.get("extra") or {}
            url = extra.get("url")
            tail = f" — {url}" if url else ""
            lines.append(f"- [{kind}] {name}{tail}")
    lines.append("")

    lines.append("=== EXPERT CHAT (most recent turns first) ===")
    if not expert_chat_turns:
        lines.append("(no prior expert conversation — work from sources + tools)")
    else:
        for turn in expert_chat_turns:
            role = turn.get("role", "user").upper()
            content = (turn.get("content") or "").strip()
            lines.append(f"[{role}]")
            lines.append(content)
            lines.append("")

    return "\n".join(lines)


def _system_instruction(format_: ScriptFormat, handoff_packet: str) -> str:
    return "\n\n".join([_BASE_RULES, _FORMAT_PROMPTS[format_], handoff_packet])


def _messages_to_contents(messages: Iterable[dict]) -> list[types.Content]:
    """Convert {role, content} entries to Gemini Content objects.

    Roles: 'user' or 'storyteller' (which maps to Gemini's 'model').
    """
    out: list[types.Content] = []
    for m in messages:
        role = "model" if m["role"] == "storyteller" else "user"
        out.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    return out


def generate_script_ideas(
    messages: list[dict],
    handoff_packet: str,
    format_: ScriptFormat,
    max_tool_rounds: int = 4,
) -> Generator[dict, None, None]:
    """Stream events from the storyteller agent.

    Event shapes (same vocabulary as the expert agent for UI symmetry):
      {"type": "text", "content": str}
      {"type": "tool_call", "name": str, "args": dict}
      {"type": "tool_result", "name": str, "result": dict}
      {"type": "citations", "citations": list[dict]}
      {"type": "done"}
    """
    client = _client()
    system_instruction = _system_instruction(format_, handoff_packet)
    contents = _messages_to_contents(messages)
    tools = [gemini_pubmed_tool()]

    for _ in range(max_tool_rounds):
        stream = client.models.generate_content_stream(
            model=STORYTELLER_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools,
            ),
        )

        function_calls: list[types.FunctionCall] = []
        assistant_parts: list[types.Part] = []

        for chunk in stream:
            if not chunk.candidates:
                continue
            cand = chunk.candidates[0]
            if cand.content and cand.content.parts:
                for part in cand.content.parts:
                    if getattr(part, "text", None):
                        assistant_parts.append(types.Part(text=part.text))
                        yield {"type": "text", "content": part.text}
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name:
                        function_calls.append(fc)
                        assistant_parts.append(types.Part(function_call=fc))
                        args = dict(fc.args) if fc.args else {}
                        yield {"type": "tool_call", "name": fc.name, "args": args}

        if assistant_parts:
            contents.append(types.Content(role="model", parts=assistant_parts))

        if not function_calls:
            yield {"type": "done"}
            return

        # Execute each custom function call and feed responses back as a user turn.
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
