from __future__ import annotations

from typing import Any, Dict, List


def get_tool_catalog() -> List[Dict[str, Any]]:
    """Static catalog used by the optimization layer for ranking and top-k selection."""
    return [
        {
            "name": "search_articles",
            "description": "Search technical articles, engineering blogs, and developer documentation.",
            "keywords": ["article", "blog", "documentation", "how to", "guide", "latest"],
            "intent_type": "search",
            "deterministic": False,
            "default_args": {"max_results": 5},
        },
        {
            "name": "search_research_papers",
            "description": "Search academic research papers and PDFs, especially from arXiv.",
            "keywords": ["research paper", "arxiv", "pdf", "study", "publication", "scientific"],
            "intent_type": "search",
            "deterministic": False,
            "default_args": {"max_results": 5},
        },
        {
            "name": "search_lit_reviews",
            "description": "Search literature reviews, survey papers, and systematic reviews.",
            "keywords": ["literature review", "survey", "systematic review", "review paper"],
            "intent_type": "search",
            "deterministic": False,
            "default_args": {"max_results": 5},
        },
        {
            "name": "translate_japanese",
            "description": "Translate text into Japanese, optionally formal tone.",
            "keywords": ["translate", "translation", "japanese", "nihongo"],
            "intent_type": "translate",
            "deterministic": True,
            "default_args": {"formal": True},
        },
        {
            "name": "translate_french",
            "description": "Translate text into French.",
            "keywords": ["translate", "translation", "french", "francais"],
            "intent_type": "translate",
            "deterministic": True,
            "default_args": {},
        },
        {
            "name": "translate_spanish",
            "description": "Translate text into Spanish.",
            "keywords": ["translate", "translation", "spanish", "espanol"],
            "intent_type": "translate",
            "deterministic": True,
            "default_args": {"dialect": "neutral"},
        },
        {
            "name": "paraphrase_formal",
            "description": "Rewrite text in a professional, formal style.",
            "keywords": ["paraphrase", "formal", "professional", "rewrite"],
            "intent_type": "paraphrase",
            "deterministic": True,
            "default_args": {},
        },
        {
            "name": "paraphrase_casual",
            "description": "Rewrite text in a casual conversational style.",
            "keywords": ["paraphrase", "casual", "conversational", "rewrite"],
            "intent_type": "paraphrase",
            "deterministic": True,
            "default_args": {},
        },
        {
            "name": "paraphrase_academic",
            "description": "Rewrite text in an academic, scholarly style.",
            "keywords": ["paraphrase", "academic", "scholarly", "rewrite"],
            "intent_type": "paraphrase",
            "deterministic": True,
            "default_args": {},
        },
    ]


def catalog_map() -> Dict[str, Dict[str, Any]]:
    return {tool["name"]: tool for tool in get_tool_catalog()}
