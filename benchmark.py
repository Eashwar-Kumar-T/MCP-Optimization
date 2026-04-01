from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from client import compare_query_modes


BASE_DIR = Path(__file__).resolve().parent


def load_queries(path: str) -> List[str]:
    queries: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("query")
            if q:
                queries.append(q)
    return queries


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


async def run_benchmark(queries: List[str], mcp_server_path: str) -> Dict[str, Any]:
    rows = []
    latency_improvements = []
    tools_reductions = []
    cost_improvements = []

    for q in queries:
        result = await compare_query_modes(q, mcp_server_path=mcp_server_path)
        comp = result["comparison"]
        latency_improvements.append(float(comp["latency_improvement_pct"]))
        tools_reductions.append(float(comp["tools_reduction_pct"]))
        cost_improvements.append(float(comp["cost_improvement_pct"]))
        rows.append(result)

    summary = {
        "queries": len(queries),
        "avg_latency_improvement_pct": round(_avg(latency_improvements), 2),
        "avg_tools_reduction_pct": round(_avg(tools_reductions), 2),
        "avg_cost_improvement_pct": round(_avg(cost_improvements), 2),
    }
    return {"summary": summary, "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A/B benchmark for MCP optimization")
    parser.add_argument("--queries", default=str(BASE_DIR / "eval_queries.jsonl"), help="Path to JSONL query file")
    parser.add_argument("--mcp-server-path", default=str(BASE_DIR / "server.py"))
    parser.add_argument("--out", default=str(BASE_DIR / "benchmark_results.json"))
    args = parser.parse_args()

    q = load_queries(args.queries)
    result = asyncio.run(run_benchmark(q, mcp_server_path=args.mcp_server_path))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result["summary"], indent=2))
