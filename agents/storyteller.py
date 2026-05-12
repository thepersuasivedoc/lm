"""Storyteller agent — Claude Opus 4.7 with web_search + pubmed_search tools."""

from __future__ import annotations

import os
from typing import Generator, Iterable, Literal

import anthropic

from config import STORYTELLER_MODEL
from agents.tools import (
    CLAUDE_PUBMED_TOOL,
    CLAUDE_WEB_SEARCH_TOOL,
    dispatch_claude_tool,
)


ScriptFormat = Literal["youtube_long", "shorts", "podcast"]


_BASE_RULES = """\
You are a creative scriptwriter collaborating with a medical student who has just
finished studying a topic with an expert tutor. Your job: turn what they learned
into compelling, accurate script ideas they can actually shoot.

GROUNDING RULES:
- The expert chat below is your primary source of truth for what the student knows
  and what claims are accurate. Quote / paraphrase from it freely.
- For factual claims NOT in the expert chat, use pubmed_search (clinical evidence)
  or web_search (general / definitional). Cite tool sources inline.
- Mark anything you couldn't verify with [VERIFY: <claim>].
- Cite format:
  - [Expert chat: <short quote or paraphrase>]
  - [PubMed: PMID <id> — <author> et al., <year>]
  - [Web: <domain> — <page title>]
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


def _client() -> anthropic.Anthropic:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")
    return anthropic.Anthropic(api_key=api_key)


def build_handoff_packet(
    expert_chat_turns: list[dict],
    sources_digest: list[dict],
    user_brief: str,
) -> str:
    """Assemble the static handoff context that gets cached on the system prompt."""
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


def _system_blocks(format_: ScriptFormat, handoff_packet: str) -> list[dict]:
    """Build system as a list of cached text blocks.

    Stable content (base rules + format prompt) goes first, then the handoff
    packet which is static within a session. The 5-min ephemeral cache makes
    follow-up requests in the same session ~90% cheaper.
    """
    fmt_prompt = _FORMAT_PROMPTS[format_]
    return [
        {
            "type": "text",
            "text": _BASE_RULES + "\n\n" + fmt_prompt,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": handoff_packet,
            "cache_control": {"type": "ephemeral"},
        },
    ]


def _to_anthropic_messages(messages: Iterable[dict]) -> list[dict]:
    """Convert {role, content} entries to Anthropic's message shape.

    Roles: 'user' or 'storyteller' (which maps to 'assistant').
    """
    out: list[dict] = []
    for m in messages:
        role = "assistant" if m["role"] == "storyteller" else "user"
        out.append({"role": role, "content": m["content"]})
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
    system_blocks = _system_blocks(format_, handoff_packet)
    convo = _to_anthropic_messages(messages)

    tools = [CLAUDE_PUBMED_TOOL, CLAUDE_WEB_SEARCH_TOOL]

    for _ in range(max_tool_rounds):
        with client.messages.stream(
            model=STORYTELLER_MODEL,
            max_tokens=16000,
            system=system_blocks,
            messages=convo,
            tools=tools,
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
        ) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    block = event.content_block
                    if block.type == "tool_use":
                        yield {
                            "type": "tool_call",
                            "name": block.name,
                            "args": {},
                        }
                elif event.type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        yield {"type": "text", "content": delta.text}

            final = stream.get_final_message()

        # Persist the assistant turn (full content list, incl. thinking + tool_use blocks).
        convo.append({"role": "assistant", "content": final.content})

        # Find any custom-tool invocations we need to execute client-side.
        # The web_search tool is server-side and resolves itself within the same response.
        custom_tool_uses = [
            b for b in final.content
            if b.type == "tool_use" and b.name in {"pubmed_search"}
        ]

        if final.stop_reason == "end_turn" and not custom_tool_uses:
            citations = _extract_citations(final.content)
            if citations:
                yield {"type": "citations", "citations": citations}
            yield {"type": "done"}
            return

        if final.stop_reason == "pause_turn" and not custom_tool_uses:
            # Server-side tool hit its iteration limit; re-send to continue.
            # Anthropic resumes automatically when the assistant turn ends with a
            # server_tool_use block — don't add a "continue" user message.
            continue

        # Execute custom tools and feed results back.
        tool_results = []
        for tool_use in custom_tool_uses:
            input_ = tool_use.input or {}
            result = dispatch_claude_tool(tool_use.name, input_)
            yield {"type": "tool_result", "name": tool_use.name, "result": result}
            is_error = "error" in result
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": _format_tool_result_content(result),
                "is_error": is_error,
            })

        convo.append({"role": "user", "content": tool_results})

    yield {"type": "text", "content": "\n\n(Stopped: tool-call loop exceeded max rounds.)"}
    yield {"type": "done"}


def _format_tool_result_content(result: dict) -> str:
    """Compact JSON-like rendering so Claude consumes it cleanly as text."""
    import json
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except Exception:
        return str(result)


def _extract_citations(content) -> list[dict]:
    """Pull citation references out of web_search server_tool_use_result blocks."""
    citations: list[dict] = []
    for block in content:
        # Web search tool results carry citation metadata on text blocks.
        if block.type == "text":
            for cit in getattr(block, "citations", []) or []:
                cite_type = getattr(cit, "type", "")
                if cite_type == "web_search_result_location":
                    citations.append({
                        "type": "web",
                        "title": getattr(cit, "title", "") or "",
                        "uri": getattr(cit, "url", "") or "",
                    })
    return citations
