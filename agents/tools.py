"""Gemini tool definitions and Python functions used by the agents.

Native file_search is configured on the agent side (per-notebook store), not here.
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
