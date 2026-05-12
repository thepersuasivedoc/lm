"""NCBI E-utilities wrapper: search PubMed + fetch abstracts."""

from __future__ import annotations

import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict

import requests

from config import (
    PUBMED_DEFAULT_RESULTS,
    PUBMED_RATE_LIMIT_NO_KEY,
    PUBMED_RATE_LIMIT_WITH_KEY,
)


BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_last_call_t = 0.0


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    authors: list[str]
    journal: str
    year: str
    abstract: str
    doi: str
    url: str


def _rate_limit_wait() -> None:
    global _last_call_t
    rate = PUBMED_RATE_LIMIT_WITH_KEY if os.environ.get("NCBI_API_KEY") else PUBMED_RATE_LIMIT_NO_KEY
    min_gap = 1.0 / rate
    elapsed = time.time() - _last_call_t
    if elapsed < min_gap:
        time.sleep(min_gap - elapsed)
    _last_call_t = time.time()


def _params(extra: dict) -> dict:
    p = {"tool": "lm-medical-tutor", "email": "user@example.com"}
    key = os.environ.get("NCBI_API_KEY")
    if key:
        p["api_key"] = key
    p.update(extra)
    return p


def search_pmids(query: str, max_results: int = PUBMED_DEFAULT_RESULTS) -> list[str]:
    _rate_limit_wait()
    r = requests.get(
        f"{BASE}/esearch.fcgi",
        params=_params({"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}),
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    return data.get("esearchresult", {}).get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[PubMedArticle]:
    if not pmids:
        return []
    _rate_limit_wait()
    r = requests.get(
        f"{BASE}/efetch.fcgi",
        params=_params({"db": "pubmed", "id": ",".join(pmids), "retmode": "xml"}),
        timeout=30,
    )
    r.raise_for_status()
    return _parse_pubmed_xml(r.text)


def _parse_pubmed_xml(xml_text: str) -> list[PubMedArticle]:
    root = ET.fromstring(xml_text)
    out: list[PubMedArticle] = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""

        # Abstract is sometimes split into AbstractText elements with labels.
        abstract_parts: list[str] = []
        for ab in article.findall(".//Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            body = "".join(ab.itertext()).strip()
            abstract_parts.append(f"{label}: {body}" if label else body)
        abstract = "\n".join(p for p in abstract_parts if p)

        authors: list[str] = []
        for au in article.findall(".//AuthorList/Author"):
            last = (au.findtext("LastName") or "").strip()
            initials = (au.findtext("Initials") or "").strip()
            if last:
                authors.append(f"{last} {initials}".strip())
            else:
                collective = (au.findtext("CollectiveName") or "").strip()
                if collective:
                    authors.append(collective)

        journal = (article.findtext(".//Journal/Title") or "").strip()
        year = (article.findtext(".//PubDate/Year") or article.findtext(".//PubDate/MedlineDate") or "").strip()[:4]

        doi = ""
        for aid in article.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi" and aid.text:
                doi = aid.text.strip()
                break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        out.append(
            PubMedArticle(
                pmid=pmid or "",
                title=title,
                authors=authors,
                journal=journal,
                year=year,
                abstract=abstract,
                doi=doi,
                url=url,
            )
        )
    return out


def search(query: str, max_results: int = PUBMED_DEFAULT_RESULTS) -> list[dict]:
    """High-level: search + fetch abstracts. Returns plain dicts (tool-call friendly)."""
    pmids = search_pmids(query, max_results=max_results)
    articles = fetch_abstracts(pmids)
    return [asdict(a) for a in articles]
