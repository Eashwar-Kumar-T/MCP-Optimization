from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class QueryMetrics:
    mode: str
    query: str
    latency_ms: float
    tools_considered: int
    selected_tools_count: int
    tool_calls_executed: int
    router_confidence: float
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    success: bool
    error_type: str
    selected_tools_json: str
    metadata_json: str
    created_at: float


class MetricsStore:
    def __init__(self, db_path: str = "runs.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mode TEXT NOT NULL,
                    query TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    tools_considered INTEGER NOT NULL,
                    selected_tools_count INTEGER NOT NULL,
                    tool_calls_executed INTEGER NOT NULL,
                    router_confidence REAL NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    success INTEGER NOT NULL,
                    error_type TEXT NOT NULL,
                    selected_tools_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def write(self, metrics: QueryMetrics) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            payload = asdict(metrics)
            conn.execute(
                """
                INSERT INTO runs (
                    mode, query, latency_ms, tools_considered, selected_tools_count, tool_calls_executed,
                    router_confidence, input_tokens, output_tokens, estimated_cost_usd, success, error_type,
                    selected_tools_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    payload["mode"],
                    payload["query"],
                    payload["latency_ms"],
                    payload["tools_considered"],
                    payload["selected_tools_count"],
                    payload["tool_calls_executed"],
                    payload["router_confidence"],
                    payload["input_tokens"],
                    payload["output_tokens"],
                    payload["estimated_cost_usd"],
                    1 if payload["success"] else 0,
                    payload["error_type"],
                    payload["selected_tools_json"],
                    payload["metadata_json"],
                    payload["created_at"],
                ),
            )
            conn.commit()
        finally:
            conn.close()


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start) * 1000.0


def extract_token_usage(payload: Any) -> Dict[str, int]:
    """Best-effort extraction across dict/object response shapes."""
    input_tokens = 0
    output_tokens = 0

    def ingest(item: Any) -> None:
        nonlocal input_tokens, output_tokens
        if not isinstance(item, dict):
            return
        input_tokens = int(item.get("prompt_tokens") or item.get("input_tokens") or input_tokens or 0)
        output_tokens = int(item.get("completion_tokens") or item.get("output_tokens") or output_tokens or 0)

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, dict):
            ingest(obj.get("token_usage", {}))
            ingest(obj.get("usage", {}))
            response_metadata = obj.get("response_metadata")
            if isinstance(response_metadata, dict):
                ingest(response_metadata.get("token_usage", response_metadata))
            for value in obj.values():
                walk(value)
            return

        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return

        for attr in ("response_metadata", "usage_metadata"):
            meta = getattr(obj, attr, None)
            if isinstance(meta, dict):
                ingest(meta.get("token_usage", meta))

        # many LangChain responses keep inner messages in `messages`
        messages = getattr(obj, "messages", None)
        if isinstance(messages, list):
            for message in messages:
                walk(message)

    walk(payload)

    if isinstance(payload, dict):
        candidates = [
            payload.get("token_usage"),
            payload.get("usage"),
            payload.get("response_metadata", {}).get("token_usage") if isinstance(payload.get("response_metadata"), dict) else None,
        ]
        for item in candidates:
            if isinstance(item, dict):
                input_tokens = int(item.get("prompt_tokens") or item.get("input_tokens") or input_tokens or 0)
                output_tokens = int(item.get("completion_tokens") or item.get("output_tokens") or output_tokens or 0)

    for attr in ("response_metadata", "usage_metadata"):
        obj = getattr(payload, attr, None)
        if isinstance(obj, dict):
            token_usage = obj.get("token_usage", obj)
            if isinstance(token_usage, dict):
                input_tokens = int(token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or input_tokens or 0)
                output_tokens = int(token_usage.get("completion_tokens") or token_usage.get("output_tokens") or output_tokens or 0)

    if input_tokens == 0 and output_tokens == 0:
        # fallback approximation to avoid always-zero cost when providers omit usage
        text = str(payload)
        approx = max(1, len(text) // 4)
        output_tokens = approx

    return {"input_tokens": input_tokens, "output_tokens": output_tokens}


def estimate_cost_usd(input_tokens: int, output_tokens: int, input_per_million: float = 0.25, output_per_million: float = 0.5) -> float:
    return round((input_tokens / 1_000_000) * input_per_million + (output_tokens / 1_000_000) * output_per_million, 8)


def make_metrics(
    *,
    mode: str,
    query: str,
    latency_ms: float,
    tools_considered: int,
    selected_tools_count: int,
    tool_calls_executed: int,
    router_confidence: float,
    token_payload: Optional[Any],
    success: bool,
    error_type: str = "",
    selected_tools: Optional[list[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> QueryMetrics:
    usage = extract_token_usage(token_payload) if token_payload is not None else {"input_tokens": 0, "output_tokens": 0}

    return QueryMetrics(
        mode=mode,
        query=query,
        latency_ms=round(latency_ms, 2),
        tools_considered=tools_considered,
        selected_tools_count=selected_tools_count,
        tool_calls_executed=tool_calls_executed,
        router_confidence=round(router_confidence, 4),
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        estimated_cost_usd=estimate_cost_usd(usage["input_tokens"], usage["output_tokens"]),
        success=success,
        error_type=error_type,
        selected_tools_json=json.dumps(selected_tools or []),
        metadata_json=json.dumps(metadata or {}),
        created_at=time.time(),
    )
