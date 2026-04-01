from mcp.server.fastmcp import FastMCP
from typing import List, Dict, Optional
import asyncio
import os

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    import arxiv
except Exception:
    arxiv = None

try:
    from googletrans import Translator
except Exception:
    Translator = None


_paraphrase_pipeline = None


def _lazy_load_paraphrase_pipeline():
    """Lazy-load paraphrase pipeline only when needed (requires transformers + torch)."""
    global _paraphrase_pipeline
    if _paraphrase_pipeline is not None:
        return _paraphrase_pipeline
    try:
        from transformers import pipeline
        _paraphrase_pipeline = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", truncation=True)
        return _paraphrase_pipeline
    except Exception as e:
        return None

mcp = FastMCP("SearchAndLanguageTools")
USE_ARXIV = os.getenv("USE_ARXIV", "0") == "1"


def _duckduckgo_search(query: str, max_results: int = 6) -> List[Dict]:
    if DDGS is None:
        return [{"title": "ddg-not-available",
                 "snippet": "duckduckgo_search package not installed",
                 "url": ""}]

    out = []
    with DDGS() as ddgs:
        for i, r in enumerate(ddgs.text(query, region="wt-wt", safesearch="Off")):
            if i >= max_results:
                break
            out.append({
                "title": r.get("title") or "",
                "snippet": r.get("body") or "",
                "url": r.get("href") or ""
            })
    return out


def _arxiv_search(query: str, max_results: int = 6) -> List[Dict]:
    """Query arXiv and return structured results. Requires arxiv package."""
    if arxiv is None:
        return [{"title": "arxiv-not-available",
                 "snippet": "arxiv package is not installed or network disabled.",
                 "url": ""}]
    results = []
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    for result in search.results():
        authors = ", ".join([a.name for a in result.authors])
        snippet = f"{result.summary[:400]}..." if result.summary else ""
        results.append({
            "title": result.title,
            "snippet": f"{authors} — {snippet}",
            "url": result.pdf_url or result.entry_id
        })
    return results


##########################################
# Search tools (article, research, reviews)
##########################################

@mcp.tool()
def search_articles(query: str, max_results: int = 6) -> List[Dict]:
    """
    Performs a Google-like search (via DuckDuckGo) for general technical articles, dev blogs, and documentation.
    Behind-the-scenes query logic: query + "blog OR article OR documentation"
    Returns list of {title, snippet, url}.
    """
    final_query = f'{query} (blog OR article OR documentation)'
    results = _duckduckgo_search(final_query, max_results=max_results)
    # Add metadata about how the query was formed
    return [{
        "tool": "search_articles",
        "final_query": final_query,
        "title": r["title"],
        "snippet": r["snippet"],
        "url": r["url"]
    } for r in results]


@mcp.tool()
def search_research_papers(query: str, max_results: int = 6) -> List[Dict]:
    """
    Targeted search for academic PDFs and scientific publications.
    Behind-the-scenes query logic: query + "filetype:pdf research paper" (or site:arxiv.org)
    Uses arXiv API primarily; falls back to DuckDuckGo PDF search.
    Returns list of {title, snippet, url}.
    """
    if USE_ARXIV:
        try:
            q_arxiv = f'{query}'
            arxiv_results = _arxiv_search(q_arxiv, max_results=max_results)
            if arxiv_results and arxiv_results[0].get("title", "") != "arxiv-not-available":
                return [{"tool": "search_research_papers",
                         "final_query": q_arxiv + " (site:arxiv.org)",
                         "title": r["title"],
                         "snippet": r["snippet"],
                         "url": r["url"]} for r in arxiv_results]
        except Exception:
            pass

    final_query = f'{query} (filetype:pdf research paper OR "research paper" OR "PDF")'
    results = _duckduckgo_search(final_query, max_results=max_results)
    return [{
        "tool": "search_research_papers",
        "final_query": final_query,
        "title": r["title"],
        "snippet": r["snippet"],
        "url": r["url"]
    } for r in results]


@mcp.tool()
def search_lit_reviews(query: str, max_results: int = 6) -> List[Dict]:
    """
    Looks for literature reviews and survey papers (comprehensive surveys).
    Query logic: query + "literature review OR survey paper"
    Returns list of {title, snippet, url}.
    """
    final_query = f'{query} ("literature review" OR "survey paper" OR "systematic review" OR "survey") filetype:pdf'
    # Prefer arXiv search for surveys too
    if arxiv and USE_ARXIV:
        # broaden query to include "survey" keywords
        arxiv_query = f'{query} AND (survey OR "literature review" OR "systematic review")'
        try:
            arxiv_results = _arxiv_search(arxiv_query, max_results=max_results)
            if arxiv_results and arxiv_results[0].get("title", "") != "arxiv-not-available":
                return [{"tool": "search_lit_reviews",
                         "final_query": arxiv_query + " (arXiv)",
                         "title": r["title"],
                         "snippet": r["snippet"],
                         "url": r["url"]} for r in arxiv_results]
        except Exception:
            pass

    # Fall back to DuckDuckGo
    results = _duckduckgo_search(final_query, max_results=max_results)
    return [{
        "tool": "search_lit_reviews",
        "final_query": final_query,
        "title": r["title"],
        "snippet": r["snippet"],
        "url": r["url"]
    } for r in results]


##########################################
# Translation tools
##########################################

def _translate_text(text: str, dest: str, prefer_formal: Optional[bool] = False) -> Dict:
    """
    Uses googletrans if available. prefer_formal is advisory: we attempt to add polite phrasing for Japanese.
    Returns {"original":..., "translated":..., "engine":"googletrans" or "fallback"}
    """
    if Translator is None:
        return {
            "original": text,
            "translated": "translation-service-not-available: googletrans not installed on server.",
            "engine": "fallback"
        }
    translator = Translator()
    to_translate = text
    if dest == "ja" and prefer_formal:
        to_translate = f"(Use polite/formal Japanese) {text}"

    try:
        r = translator.translate(to_translate, dest=dest)
        if asyncio.iscoroutine(r):
            r = asyncio.run(r)
        return {"original": text, "translated": r.text, "engine": "googletrans"}
    except Exception as e:
        return {"original": text, "translated": f"translation-failed: {e}", "engine": "googletrans"}


@mcp.tool()
def translate_japanese(text: str, formal: bool = True) -> Dict:
    """
    Translates text into natural Japanese.
    If formal=True, attempts to use polite/Kejgo tone.
    """
    return _translate_text(text, dest="ja", prefer_formal=formal)


@mcp.tool()
def translate_french(text: str) -> Dict:
    """Translates text into French (standard audience)."""
    return _translate_text(text, dest="fr", prefer_formal=False)


@mcp.tool()
def translate_spanish(text: str, dialect: str = "neutral") -> Dict:
    """
    Translates text into Spanish. Defaults to neutral Spanish unless dialect specified.
    dialect param is advisory and not strictly enforced by googletrans.
    """
    return _translate_text(text, dest="es", prefer_formal=False)


##########################################
# Paraphrasing tools
##########################################

def _paraphrase_with_model(text: str, task_style: str, num_return: int = 1) -> List[str]:
    """
    Use transformers pipeline (if available) to paraphrase. task_style is advisory (formal/casual/academic)
    Falls back to simple rule-based paraphrasing if transformers/torch unavailable.
    Returns list of paraphrases.
    """
    pipeline = _lazy_load_paraphrase_pipeline()
    if pipeline is None:
        if task_style == "formal":
            return [text.replace("don't", "do not").replace("we're", "we are")]
        if task_style == "casual":
            return [text.replace("do not", "don't").replace("cannot", "can't")]
        if task_style == "academic":
            return [f"In academic terms: {text}"]
        return [text]

    prompt = f"paraphrase: {text} style: {task_style}"
    try:
        gen = pipeline(prompt, max_length=256, num_return_sequences=num_return)
        return [g["generated_text"].strip() for g in gen]
    except Exception:
        return [text]


@mcp.tool()
def paraphrase_formal(text: str) -> Dict:
    """Rewrites text to be professional and authoritative."""
    out = _paraphrase_with_model(text, task_style="formal", num_return=1)[0]
    return {"original": text, "paraphrased": out, "style": "formal"}


@mcp.tool()
def paraphrase_casual(text: str) -> Dict:
    """Rewrites text to be conversational and relaxed."""
    out = _paraphrase_with_model(text, task_style="casual", num_return=1)[0]
    return {"original": text, "paraphrased": out, "style": "casual"}


@mcp.tool()
def paraphrase_academic(text: str) -> Dict:
    """Rewrites text for a scholarly context."""
    out = _paraphrase_with_model(text, task_style="academic", num_return=1)[0]
    return {"original": text, "paraphrased": out, "style": "academic"}


if __name__ == "__main__":
    mcp.run(transport="stdio")
