from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from benchmark import load_queries, run_benchmark
from client import compare_query_modes


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_PATH = str(BASE_DIR / "server.py")
DEFAULT_DB_PATH = str(BASE_DIR / "runs.db")

st.set_page_config(page_title="MCP Optimization Dashboard", layout="wide")


def _get_metric(metrics: Dict[str, Any], key: str, default: Any = 0) -> Any:
    return metrics.get(key, default) if isinstance(metrics, dict) else default


def _fmt_cost(value: Any) -> str:
    try:
        return f"${float(value):.6f}"
    except Exception:
        return "$0.000000"


def _fmt_ms(value: Any) -> str:
    try:
        return f"{float(value):,.0f} ms"
    except Exception:
        return "0 ms"


def _render_answer_panel(title: str, payload: Dict[str, Any], accent: str) -> None:
    metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    answer = payload.get("answer") if isinstance(payload, dict) else None
    answer = answer if isinstance(answer, str) and answer.strip() else "(no answer)"
    routing = payload.get("routing", "unknown") if isinstance(payload, dict) else "unknown"
    error = _get_metric(metrics, "error_type", "")
    success = bool(_get_metric(metrics, "success", False))
    with st.container(border=True):
        head_left, head_right = st.columns([4, 1])
        with head_left:
            st.markdown(f"### {title}")
            st.caption(f"Routing mode: {routing}")
        with head_right:
            if success:
                st.success("Success", icon="✅")
            else:
                st.error(error or "Failed", icon="⚠️")

        metric_cols = st.columns(4)
        metric_cols[0].metric("Latency", _fmt_ms(_get_metric(metrics, "latency_ms", 0)), "end-to-end request time")
        metric_cols[1].metric("Tool Calls", str(_get_metric(metrics, "tool_calls_executed", 0)), "actual executed tool calls")
        metric_cols[2].metric("Cost", _fmt_cost(_get_metric(metrics, "estimated_cost_usd", 0)), "estimated model cost")
        metric_cols[3].metric("Tools Considered", str(_get_metric(metrics, "tools_considered", 0)), "tool context exposed")

        st.markdown("**Final answer**")
        st.write(answer)

        with st.expander(f"Show raw payload for {title.lower()}", expanded=False):
            st.json(payload)


st.markdown(
    """
    <style>
        .hero {
            padding: 1.25rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(86,115,255,0.15), rgba(0,194,255,0.10), rgba(106,255,184,0.10));
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 10px 30px rgba(0,0,0,0.12);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            font-size: 0.98rem;
            opacity: 0.82;
        }
        .delta-good { color: #34d399; font-weight: 700; }
        .delta-bad { color: #fb7185; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-title">MCP Optimization Dashboard</div>
        <div class="hero-subtitle">Compare the naive baseline against the optimized router with stacked results, color-coded metrics, and benchmark history.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    server_path = st.text_input("MCP server path", DEFAULT_SERVER_PATH)
    query = st.text_area("Query", "Find recent research papers about MCP optimization", height=120)
    run_compare = st.button("Run baseline vs optimized", use_container_width=True)
    run_benchmark_btn = st.button("Run full benchmark", use_container_width=True)

st.subheader("Live Comparison")
st.caption("Baseline is shown first. Optimized is shown underneath for direct comparison.")

comparison_result = None
benchmark_result = None

if run_compare:
    with st.spinner("Running comparison..."):
        comparison_result = asyncio.run(compare_query_modes(query, mcp_server_path=server_path))

if comparison_result:
    comp = comparison_result.get("comparison", {})
    comp_cols = st.columns(3)
    comp_cols[0].metric(
        "Latency improvement",
        f"{comp.get('latency_improvement_pct', 0):+.2f}%",
        "optimized vs baseline",
    )
    comp_cols[1].metric(
        "Tool-call change",
        f"{comp.get('tools_reduction_pct', 0):+.2f}%",
        "lower is better",
    )
    comp_cols[2].metric(
        "Cost improvement",
        f"{comp.get('cost_improvement_pct', 0):+.2f}%",
        "estimated model cost",
    )

    st.markdown("### Baseline")
    _render_answer_panel("Baseline", comparison_result.get("baseline", {}), "baseline")

    st.markdown("### Optimized")
    _render_answer_panel("Optimized", comparison_result.get("optimized", {}), "optimized")

if run_benchmark_btn:
    with st.spinner("Benchmark running..."):
        queries = load_queries(str(BASE_DIR / "eval_queries.jsonl"))
        benchmark_result = asyncio.run(run_benchmark(queries, mcp_server_path=server_path))

if benchmark_result:
    st.subheader("Benchmark Summary")
    summary = benchmark_result.get("summary", {})
    summary_cols = st.columns(3)
    summary_cols[0].metric("Avg latency improvement", f"{summary.get('avg_latency_improvement_pct', 0):.2f}%")
    summary_cols[1].metric("Avg tool reduction", f"{summary.get('avg_tools_reduction_pct', 0):.2f}%")
    summary_cols[2].metric("Avg cost improvement", f"{summary.get('avg_cost_improvement_pct', 0):.2f}%")

    detail_rows = []
    for row in benchmark_result.get("rows", []):
        detail_rows.append(
            {
                "query": row["query"],
                "latency_improvement_pct": row["comparison"]["latency_improvement_pct"],
                "tools_reduction_pct": row["comparison"]["tools_reduction_pct"],
                "cost_improvement_pct": row["comparison"]["cost_improvement_pct"],
            }
        )
    df = pd.DataFrame(detail_rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

st.subheader("Run History")
if Path(DEFAULT_DB_PATH).exists():
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    df = pd.read_sql_query(
        "SELECT mode, query, latency_ms, tools_considered, selected_tools_count, tool_calls_executed, estimated_cost_usd, success, error_type, created_at FROM runs ORDER BY id DESC LIMIT 200",
        conn,
    )
    conn.close()

    if not df.empty:
        display_df = df.copy()
        display_df["latency_ms"] = display_df["latency_ms"].map(lambda x: f"{x:,.0f}")
        display_df["estimated_cost_usd"] = display_df["estimated_cost_usd"].map(lambda x: f"${x:.6f}")
        display_df["success"] = display_df["success"].map(lambda x: "yes" if x else "no")
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No runs logged yet. Execute at least one query.")
else:
    st.info("No runs logged yet. Execute at least one query.")
