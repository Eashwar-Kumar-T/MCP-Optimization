from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class OptimizerConfig:
    top_k_min: int = 1
    top_k_max: int = 3
    high_conf_threshold: float = 0.78
    medium_conf_threshold: float = 0.45


_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "for", "and", "or", "in", "on", "with",
    "about", "please", "can", "you", "me", "this", "that", "it", "into", "as", "be"
}


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\-]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS]


def _cosine_from_counters(a: Counter, b: Counter) -> float:
    common = set(a).intersection(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _intent_text(tool: Dict[str, Any]) -> str:
    return f"{tool['name']} {tool['description']} {' '.join(tool.get('keywords', []))}"


def fast_intent_gate(query: str) -> Dict[str, Any]:
    q = query.strip().lower()

    if any(k in q for k in ["translate", "translation"]):
        if "japanese" in q:
            return {"forced_tools": ["translate_japanese"], "needs_tools": True, "reason": "translation intent"}
        if "french" in q:
            return {"forced_tools": ["translate_french"], "needs_tools": True, "reason": "translation intent"}
        if "spanish" in q:
            return {"forced_tools": ["translate_spanish"], "needs_tools": True, "reason": "translation intent"}

    if "paraphrase" in q or "rewrite" in q:
        if "academic" in q:
            return {"forced_tools": ["paraphrase_academic"], "needs_tools": True, "reason": "paraphrase intent"}
        if "casual" in q:
            return {"forced_tools": ["paraphrase_casual"], "needs_tools": True, "reason": "paraphrase intent"}
        return {"forced_tools": ["paraphrase_formal"], "needs_tools": True, "reason": "paraphrase intent"}

    search_terms = [
        "find", "search", "latest", "papers", "research", "survey", "literature review", "sources", "references"
    ]
    if any(t in q for t in search_terms):
        return {"forced_tools": [], "needs_tools": True, "reason": "search intent"}

    direct_terms = ["explain", "what is", "why", "difference", "compare", "example", "define"]
    if any(t in q for t in direct_terms):
        return {"forced_tools": [], "needs_tools": False, "reason": "likely answerable directly"}

    return {"forced_tools": [], "needs_tools": True, "reason": "uncertain intent"}


def semantic_rank_tools(query: str, catalog: List[Dict[str, Any]]) -> List[Tuple[str, float]]:
    q_counter = Counter(_tokenize(query))
    ranked: List[Tuple[str, float]] = []
    for tool in catalog:
        t_counter = Counter(_tokenize(_intent_text(tool)))
        ranked.append((tool["name"], _cosine_from_counters(q_counter, t_counter)))
    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked


def select_top_k(ranked: List[Tuple[str, float]], config: OptimizerConfig) -> Dict[str, Any]:
    if not ranked:
        return {"selected": [], "confidence": 0.0, "k": 0}

    top_score = ranked[0][1]
    if top_score >= config.high_conf_threshold:
        k = config.top_k_min
    elif top_score >= config.medium_conf_threshold:
        k = min(2, config.top_k_max)
    else:
        k = config.top_k_max

    selected = [name for name, score in ranked[:k] if score > 0]
    if not selected:
        selected = [ranked[0][0]]
    return {"selected": selected, "confidence": top_score, "k": len(selected)}


def _extract_text_payload(query: str) -> str:
    q = query.strip()
    separators = [":", "\"", "'", " - "]
    for sep in separators:
        if sep in q:
            parts = q.split(sep, 1)
            candidate = parts[1].strip().strip('"').strip("'")
            if candidate:
                return candidate
    return q


def build_tool_args(query: str, selected_tools: List[str], catalog_map: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    args: Dict[str, Dict[str, Any]] = {}
    text_payload = _extract_text_payload(query)

    for tool in selected_tools:
        default_args = dict(catalog_map.get(tool, {}).get("default_args", {}))

        if tool.startswith("search_"):
            args[tool] = {"query": query, **default_args}
        elif tool.startswith("translate_"):
            args[tool] = {"text": text_payload, **default_args}
        elif tool.startswith("paraphrase_"):
            args[tool] = {"text": text_payload, **default_args}
        else:
            args[tool] = default_args
    return args


def build_optimized_plan(query: str, catalog: List[Dict[str, Any]], config: OptimizerConfig | None = None) -> Dict[str, Any]:
    config = config or OptimizerConfig()
    gate = fast_intent_gate(query)

    if gate["needs_tools"] is False:
        return {
            "needs_tools": False,
            "selected_tools": [],
            "tool_args": {},
            "confidence": 1.0,
            "rationale": gate["reason"],
            "deterministic": False,
            "shortlist_scores": [],
        }

    if gate["forced_tools"]:
        selected_tools = gate["forced_tools"]
        score = 0.95
        shortlist_scores = [(selected_tools[0], score)]
    else:
        ranked = semantic_rank_tools(query, catalog)
        selection = select_top_k(ranked, config)
        selected_tools = selection["selected"]
        score = selection["confidence"]
        shortlist_scores = ranked[: config.top_k_max]

    name_to_tool = {t["name"]: t for t in catalog}
    deterministic = len(selected_tools) == 1 and bool(name_to_tool.get(selected_tools[0], {}).get("deterministic"))

    return {
        "needs_tools": True,
        "selected_tools": selected_tools,
        "tool_args": build_tool_args(query, selected_tools, name_to_tool),
        "confidence": score,
        "rationale": gate["reason"],
        "deterministic": deterministic,
        "shortlist_scores": shortlist_scores,
    }
