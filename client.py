import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from metrics import MetricsStore, Timer, make_metrics
from optimizer import build_optimized_plan
from tool_catalog import get_tool_catalog

load_dotenv(override=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY in your environment (.env)")

AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "60"))
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "20"))

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SERVER_PATH = str(BASE_DIR / "server.py")
MCP_SERVER_PATH = os.getenv("MCP_SERVER_PATH", DEFAULT_SERVER_PATH)

ROUTER_MODEL_CONFIG = {
    "model": "openai/gpt-oss-20b",
    "api_key": GROQ_API_KEY,
    "temperature": 0.0,
    "model_kwargs": {"tool_choice": "none"},
}

AGENT_MODEL_CONFIG = {
    "model": "openai/gpt-oss-20b",
    "api_key": GROQ_API_KEY,
    "temperature": 0.1,
    "model_kwargs": {"tool_choice": "auto"},
}

ROUTER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_tools": {"type": "boolean"},
        "selected_tools": {"type": "array", "items": {"type": "string"}},
        "tool_args": {"type": "object", "additionalProperties": True},
        "rationale": {"type": "string"},
    },
    "required": ["needs_tools", "selected_tools", "tool_args", "rationale"],
}


def _message_to_text(obj: Any) -> str:
    if isinstance(obj, dict):
        return str(obj.get("output") or obj.get("text") or obj)
    content = getattr(obj, "content", None)
    if isinstance(content, list):
        return "\n".join(str(c) for c in content)
    if content:
        return str(content)
    return str(obj)


def _extract_final_answer(obj: Any) -> str:
    if obj is None:
        return ""

    if isinstance(obj, dict):
        for key in ("answer", "output", "text"):
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        messages = obj.get("messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                answer = _extract_final_answer(message)
                if answer:
                    return answer
        return ""

    content = getattr(obj, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str) and text.strip():
                    chunks.append(text.strip())
            elif isinstance(item, str) and item.strip():
                chunks.append(item.strip())
        if chunks:
            return "\n".join(chunks).strip()

    if hasattr(obj, "messages"):
        messages = getattr(obj, "messages")
        if isinstance(messages, list):
            for message in reversed(messages):
                answer = _extract_final_answer(message)
                if answer:
                    return answer

    return ""


def make_router_prompt(user_query: str, available_tools: List[str]) -> str:
    return f"""
You are a tool router. Return JSON only and match this schema:
{json.dumps(ROUTER_JSON_SCHEMA, indent=2)}

Available tools: {available_tools}

Rules:
- Use the minimum number of tools.
- If no tool is needed, set needs_tools=false, selected_tools=[], tool_args={{}}.
- selected_tools must only contain names from Available tools.

User query:
{user_query}
""".strip()


def _parse_json_from_text(raw_text: str) -> Dict[str, Any]:
    start = raw_text.find("{")
    if start == -1:
        raise ValueError("Router model returned no JSON object")

    for end in range(len(raw_text), start, -1):
        chunk = raw_text[start:end]
        try:
            parsed = json.loads(chunk)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    raise ValueError(f"Failed to parse router JSON: {raw_text}")


async def ask_router_model(query: str, available_tools: List[str]) -> Dict[str, Any]:
    model = ChatGroq(**ROUTER_MODEL_CONFIG)
    resp = await model.ainvoke([
        HumanMessage(content=make_router_prompt(query, available_tools))
    ])
    parsed = _parse_json_from_text(_message_to_text(resp))

    parsed.setdefault("needs_tools", False)
    parsed.setdefault("selected_tools", [])
    parsed.setdefault("tool_args", {})
    parsed.setdefault("rationale", "")

    if not isinstance(parsed["tool_args"], dict):
        parsed["tool_args"] = {}

    parsed["selected_tools"] = [t for t in parsed["selected_tools"] if t in available_tools]
    return parsed


async def _answer_direct(query: str) -> Dict[str, Any]:
    model = ChatGroq(**AGENT_MODEL_CONFIG)
    response = await model.ainvoke([
        SystemMessage(content="Answer clearly and concisely."),
        HumanMessage(content=query),
    ])
    return {"text": _extract_final_answer(response) or _message_to_text(response), "raw": response}


async def _summarize_tool_outputs(query: str, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
    model = ChatGroq(**AGENT_MODEL_CONFIG)
    response = await model.ainvoke([
        SystemMessage(content="Summarize tool outputs into a final user answer. Mention key sources when links exist."),
        HumanMessage(content=f"Query: {query}\n\nTool outputs:\n{json.dumps(outputs, indent=2, default=str)}"),
    ])
    return {"text": _extract_final_answer(response) or _message_to_text(response), "raw": response}


def _count_tool_calls(raw_response: Any) -> int:
    count = 0

    def walk(obj: Any) -> None:
        nonlocal count
        if obj is None:
            return
        if isinstance(obj, dict):
            if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
                count += len(obj["tool_calls"])
            for value in obj.values():
                walk(value)
            return
        if isinstance(obj, list):
            for item in obj:
                walk(item)
            return

        tool_calls = getattr(obj, "tool_calls", None)
        if isinstance(tool_calls, list):
            count += len(tool_calls)

        messages = getattr(obj, "messages", None)
        if isinstance(messages, list):
            for m in messages:
                walk(m)

    walk(raw_response)
    return count


async def _run_agent(query: str, tools: List[Any], tool_args: Dict[str, Any]) -> Dict[str, Any]:
    agent_model = ChatGroq(**AGENT_MODEL_CONFIG)
    agent = create_agent(agent_model, tools=tools)
    instruction = (
        "Use tools only when needed. If tool arguments are provided, prefer them. "
        "Return a concise final answer."
    )
    if tool_args:
        instruction += f"\nTool arguments: {json.dumps(tool_args, indent=2)}"

    response = await asyncio.wait_for(
        agent.ainvoke({
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": query},
            ]
        }),
        timeout=AGENT_TIMEOUT_SECONDS,
    )
    return {"text": _extract_final_answer(response) or _message_to_text(response), "raw": response}


async def _execute_tools_direct(name_to_tool: Dict[str, Any], selected_tools: List[str], tool_args: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for tool_name in selected_tools:
        tool = name_to_tool.get(tool_name)
        if tool is None:
            outputs.append({"tool": tool_name, "error": "tool-not-available"})
            continue

        args = tool_args.get(tool_name, {})
        try:
            if hasattr(tool, "ainvoke"):
                result = await asyncio.wait_for(tool.ainvoke(args), timeout=TOOL_TIMEOUT_SECONDS)
            else:
                result = tool.invoke(args)
            outputs.append({"tool": tool_name, "args": args, "result": result})
        except asyncio.TimeoutError:
            outputs.append({"tool": tool_name, "args": args, "error": f"tool-timeout-{TOOL_TIMEOUT_SECONDS}s"})
        except Exception as exc:
            outputs.append({"tool": tool_name, "args": args, "error": str(exc)})
    return outputs


async def handle_user_query(
    user_query: str,
    *,
    mode: str = "optimized",
    mcp_server_path: str = MCP_SERVER_PATH,
    metrics_store: MetricsStore | None = None,
) -> Dict[str, Any]:
    metrics_store = metrics_store or MetricsStore(str(BASE_DIR / "runs.db"))
    timer = Timer()
    catalog = get_tool_catalog()
    available_catalog_tool_names = [t["name"] for t in catalog]

    tool_calls_executed = 0
    tools_considered = len(available_catalog_tool_names)
    selected_tools: List[str] = []
    router_confidence = 0.0
    token_payload: Any = None
    success = True
    error_type = ""
    response_payload: Dict[str, Any]

    try:
        server_params = StdioServerParameters(command="python", args=[mcp_server_path])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                name_to_tool = {t.name: t for t in tools}
                live_tool_names = list(name_to_tool.keys())

                if mode == "baseline":
                    try:
                        router = await ask_router_model(user_query, live_tool_names)
                    except Exception as router_exc:
                        fallback_plan = build_optimized_plan(user_query, catalog)
                        fallback_tools = [t for t in fallback_plan.get("selected_tools", []) if t in live_tool_names]
                        router = {
                            "needs_tools": bool(fallback_tools),
                            "selected_tools": fallback_tools,
                            "tool_args": fallback_plan.get("tool_args", {}),
                            "rationale": f"router fallback: {type(router_exc).__name__}",
                        }

                    selected_tools = router.get("selected_tools", [])
                    router_confidence = 0.5

                    if not router.get("needs_tools", False):
                        direct = await _answer_direct(user_query)
                        token_payload = direct["raw"]
                        response_payload = {
                            "mode": "baseline",
                            "routing": "direct",
                            "router": router,
                            "answer": direct["text"],
                        }
                    else:
                        if selected_tools:
                            baseline_tools = [name_to_tool[t] for t in selected_tools if t in name_to_tool]
                        else:
                            baseline_tools = list(name_to_tool.values())

                        tools_considered = len(baseline_tools)
                        try:
                            agent_out = await _run_agent(user_query, baseline_tools, router.get("tool_args", {}))
                            tool_calls_executed = max(_count_tool_calls(agent_out.get("raw")), len(selected_tools))
                            token_payload = agent_out["raw"]
                            response_payload = {
                                "mode": "baseline",
                                "routing": "llm_with_all_tools",
                                "router": router,
                                "answer": agent_out["text"],
                            }
                        except Exception as agent_exc:
                            fallback_tools = selected_tools[:1] if selected_tools else []
                            outputs = await _execute_tools_direct(name_to_tool, fallback_tools, router.get("tool_args", {}))
                            tool_calls_executed = len(outputs)
                            summarized = await _summarize_tool_outputs(user_query, outputs)
                            token_payload = summarized["raw"]
                            response_payload = {
                                "mode": "baseline",
                                "routing": "direct_tool_fallback",
                                "router": router,
                                "agent_error": str(agent_exc),
                                "tool_outputs": outputs,
                                "answer": summarized["text"],
                            }
                else:
                    plan = build_optimized_plan(user_query, catalog)
                    router_confidence = float(plan.get("confidence", 0.0))
                    selected_tools = [t for t in plan.get("selected_tools", []) if t in live_tool_names]
                    tools_considered = len(selected_tools) if selected_tools else 0

                    if not plan.get("needs_tools", False) or not selected_tools:
                        direct = await _answer_direct(user_query)
                        token_payload = direct["raw"]
                        response_payload = {
                            "mode": "optimized",
                            "routing": "direct",
                            "plan": plan,
                            "answer": direct["text"],
                        }
                    elif plan.get("deterministic", False):
                        outputs = await _execute_tools_direct(name_to_tool, selected_tools, plan.get("tool_args", {}))
                        tool_calls_executed = len(outputs)
                        summarized = await _summarize_tool_outputs(user_query, outputs)
                        token_payload = summarized["raw"]
                        response_payload = {
                            "mode": "optimized",
                            "routing": "deterministic_early_exit",
                            "plan": plan,
                            "tool_outputs": outputs,
                            "answer": summarized["text"],
                        }
                    else:
                        shortlisted_tools = [name_to_tool[t] for t in selected_tools if t in name_to_tool]
                        try:
                            agent_out = await _run_agent(user_query, shortlisted_tools, plan.get("tool_args", {}))
                            tool_calls_executed = max(_count_tool_calls(agent_out.get("raw")), len(selected_tools))
                            token_payload = agent_out["raw"]
                            response_payload = {
                                "mode": "optimized",
                                "routing": "llm_with_shortlist",
                                "plan": plan,
                                "answer": agent_out["text"],
                            }
                        except Exception as agent_exc:
                            outputs = await _execute_tools_direct(name_to_tool, selected_tools, plan.get("tool_args", {}))
                            tool_calls_executed = len(outputs)
                            summarized = await _summarize_tool_outputs(user_query, outputs)
                            token_payload = summarized["raw"]
                            response_payload = {
                                "mode": "optimized",
                                "routing": "direct_tool_fallback",
                                "plan": plan,
                                "agent_error": str(agent_exc),
                                "tool_outputs": outputs,
                                "answer": summarized["text"],
                            }
    except Exception as exc:
        success = False
        error_type = type(exc).__name__
        try:
            direct = await _answer_direct(user_query)
            token_payload = direct["raw"]
            success = True
            error_type = f"recovered:{type(exc).__name__}"
            response_payload = {
                "mode": mode,
                "routing": "error_direct_fallback",
                "error": str(exc),
                "answer": direct["text"],
            }
        except Exception:
            response_payload = {
                "mode": mode,
                "error": str(exc),
                "answer": "Unable to complete tool workflow; please try again with a narrower query.",
            }

    latency_ms = timer.elapsed_ms()
    metrics = make_metrics(
        mode=mode,
        query=user_query,
        latency_ms=latency_ms,
        tools_considered=tools_considered,
        selected_tools_count=len(selected_tools),
        tool_calls_executed=tool_calls_executed,
        router_confidence=router_confidence,
        token_payload=token_payload,
        success=success,
        error_type=error_type,
        selected_tools=selected_tools,
        metadata={"mcp_server_path": mcp_server_path},
    )
    metrics_store.write(metrics)

    response_payload["metrics"] = {
        "latency_ms": metrics.latency_ms,
        "tools_considered": metrics.tools_considered,
        "selected_tools_count": metrics.selected_tools_count,
        "tool_calls_executed": metrics.tool_calls_executed,
        "router_confidence": metrics.router_confidence,
        "estimated_cost_usd": metrics.estimated_cost_usd,
        "success": metrics.success,
        "error_type": metrics.error_type,
    }
    return response_payload


async def compare_query_modes(user_query: str, mcp_server_path: str = MCP_SERVER_PATH) -> Dict[str, Any]:
    metrics_store = MetricsStore(str(BASE_DIR / "runs.db"))
    baseline = await handle_user_query(
        user_query,
        mode="baseline",
        mcp_server_path=mcp_server_path,
        metrics_store=metrics_store,
    )
    optimized = await handle_user_query(
        user_query,
        mode="optimized",
        mcp_server_path=mcp_server_path,
        metrics_store=metrics_store,
    )

    b_lat = float(baseline.get("metrics", {}).get("latency_ms", 0.0))
    o_lat = float(optimized.get("metrics", {}).get("latency_ms", 0.0))
    b_tools = int(baseline.get("metrics", {}).get("tool_calls_executed", 0))
    o_tools = int(optimized.get("metrics", {}).get("tool_calls_executed", 0))
    b_cost = float(baseline.get("metrics", {}).get("estimated_cost_usd", 0.0))
    o_cost = float(optimized.get("metrics", {}).get("estimated_cost_usd", 0.0))

    latency_improvement_pct = ((b_lat - o_lat) / b_lat * 100.0) if b_lat > 0 else 0.0
    tools_reduction_pct = ((b_tools - o_tools) / b_tools * 100.0) if b_tools > 0 else 0.0
    cost_improvement_pct = ((b_cost - o_cost) / b_cost * 100.0) if b_cost > 0 else 0.0

    return {
        "query": user_query,
        "baseline": baseline,
        "optimized": optimized,
        "comparison": {
            "latency_improvement_pct": round(latency_improvement_pct, 2),
            "tools_reduction_pct": round(tools_reduction_pct, 2),
            "cost_improvement_pct": round(cost_improvement_pct, 2),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP baseline/optimized router with metrics")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--mode", choices=["baseline", "optimized", "compare"], default="optimized")
    parser.add_argument("--mcp-server-path", type=str, default=MCP_SERVER_PATH)
    args = parser.parse_args()

    async def _main() -> None:
        if args.mode == "compare":
            out = await compare_query_modes(args.query, mcp_server_path=args.mcp_server_path)
        else:
            out = await handle_user_query(args.query, mode=args.mode, mcp_server_path=args.mcp_server_path)
        print(json.dumps(out, indent=2, default=str))

    asyncio.run(_main())
