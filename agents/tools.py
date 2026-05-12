"""Tool definitions for both providers.

The same Python functions are exposed:
  - To Gemini, via google.genai.types.Tool + FunctionDeclaration.
  - To Claude, via the Anthropic tool-use schema.

Tool functions live here. Native tools (Gemini file_search/google_search,
Claude web_search) are configured on the agent side, not here.
"""

from __future__ import annotations

from google.genai import types

from ingestion import pubmed


# ---------- Underlying Python functions ----------

def pubmed_search(query: str, max_results: int = 5) -> dict:
    """Search PubMed for biomedical literature and return abstracts.

    Use when uploaded sources do not cover the topic and you need
    peer-reviewed medical evidence. Returns a dict with a list of articles
    (pmid, title, authors, journal, year, abstract, doi, url).
    """
    if max_results < 1:
        max_results = 1
    if max_results > 20:
        max_results = 20
    articles = pubmed.search(query=query, max_results=max_results)
    return {"articles": articles, "count": len(articles)}


# ---------- Gemini-side tool declarations ----------

_pubmed_function_declaration = types.FunctionDeclaration(
    name="pubmed_search",
    description=(
        "Search PubMed (NCBI E-utilities) for peer-reviewed biomedical literature. "
        "Use this only when the user's uploaded sources don't cover the topic and "
        "you need authoritative medical evidence."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Search query, ideally specific medical terms (e.g. 'left ventricular hypertrophy pathophysiology').",
            ),
            "max_results": types.Schema(
                type=types.Type.INTEGER,
                description="Maximum results to return. Default 5, max 20.",
            ),
        },
        required=["query"],
    ),
)


def gemini_pubmed_tool() -> types.Tool:
    return types.Tool(function_declarations=[_pubmed_function_declaration])


def gemini_google_search_tool() -> types.Tool:
    return types.Tool(google_search=types.GoogleSearch())


# Dispatch helper for Gemini function-call responses.
GEMINI_FUNCTION_DISPATCH = {
    "pubmed_search": pubmed_search,
}


def dispatch_gemini_function_call(name: str, args: dict) -> dict:
    fn = GEMINI_FUNCTION_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown function: {name}"}
    try:
        return fn(**args)
    except Exception as e:
        return {"error": str(e)}


# ---------- Claude-side tool schemas ----------

CLAUDE_PUBMED_TOOL = {
    "name": "pubmed_search",
    "description": (
        "Search PubMed (NCBI E-utilities) for peer-reviewed biomedical literature. "
        "Use to verify clinical claims or find authoritative sources for script material."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query, ideally specific medical terms.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum results to return. Default 5, max 20.",
            },
        },
        "required": ["query"],
    },
}


# Anthropic's server-side web search tool (current version with dynamic filtering).
CLAUDE_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "max_uses": 5,
}


CLAUDE_FUNCTION_DISPATCH = {
    "pubmed_search": pubmed_search,
}


def dispatch_claude_tool(name: str, input_: dict) -> dict:
    fn = CLAUDE_FUNCTION_DISPATCH.get(name)
    if fn is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return fn(**input_)
    except Exception as e:
        return {"error": str(e)}
