# MCP Optimization

MCP Optimization is a baseline-vs-optimized tool-routing application for MCP-enabled LLM workflows.

It provides:

- a **baseline pipeline** (naive / broad tool usage)
- an **optimized pipeline** (intent gate + semantic ranking + top-k tool selection)
- side-by-side comparison in CLI and Streamlit UI
- persisted telemetry (`runs.db`) for latency, tool usage, cost estimates, and reliability

---

## 1) Problem this project solves

In naive tool-calling systems, the model often sees too many tools and may over-call them. This increases:

- response latency
- tool-call count
- model context overhead
- overall inference cost

This project introduces a practical routing layer that keeps correctness while reducing unnecessary tool exposure.

---

## 2) End-to-end architecture

```mermaid
flowchart TD
		U[User Query] --> C[client.py Orchestrator]
		C --> M{Mode}

		M -->|baseline| B1[LLM Router]
		B1 --> B2[Tool selection]
		B2 --> B3[Agent with MCP tools]

		M -->|optimized| O1[fast_intent_gate]
		O1 --> O2[semantic_rank_tools]
		O2 --> O3[select_top_k]
		O3 --> O4{deterministic?}
		O4 -->|yes| O5[Direct tool execution]
		O4 -->|no| O6[Agent with shortlisted tools]

		B3 --> F[Final answer]
		O5 --> F
		O6 --> F

		F --> T[metrics.py telemetry]
		T --> DB[(runs.db)]
		DB --> UI[app.py Streamlit Dashboard]
		DB --> BENCH[benchmark.py]
```

---

## 3) Repository structure

- `client.py`  
	Main async orchestrator. Supports modes:
	- `baseline`
	- `optimized`
	- `compare`

- `server.py`  
	MCP server exposing search, translation, and paraphrasing tools.

- `optimizer.py`  
	Optimization policy logic:
	- `fast_intent_gate()`
	- `semantic_rank_tools()`
	- `select_top_k()`
	- `build_optimized_plan()`

- `tool_catalog.py`  
	Tool metadata used for ranking, defaults, and deterministic eligibility.

- `metrics.py`  
	Telemetry model + SQLite writer + token/cost extraction.

- `benchmark.py`  
	Runs A/B evaluation using queries from `eval_queries.jsonl`.

- `app.py`  
	Streamlit UI for side-by-side comparison, benchmark summary, and run history.

- `eval_queries.jsonl`  
	Benchmark query set.

- `requirements.txt`  
	Runtime dependencies. (`transformers`/`torch` path is optional in current setup.)

---

## 4) Baseline flow (current implementation)

```mermaid
flowchart TD
		Q[Query] --> R[ask_router_model]
		R --> D{needs_tools?}
		D -->|no| A1[Direct LLM answer]
		D -->|yes| A2[Agent + selected/all available tools]
		A2 --> A3{Agent failure/timeout?}
		A3 -->|yes| A4[Direct tool fallback + summarization]
		A3 -->|no| A5[Agent final answer]
		A1 --> OUT[Response payload + metrics]
		A4 --> OUT
		A5 --> OUT
```

Notes:

- Router failure falls back to a safe plan.
- Agent and tool calls are timeout-protected.
- Error fallback still returns a user-facing answer when possible.

---

## 5) Optimized flow (key logic)

```mermaid
flowchart TD
		Q[Query] --> G[fast_intent_gate]
		G --> H{Needs tools?}
		H -->|no| D1[Direct LLM answer]
		H -->|yes| R[semantic_rank_tools]
		R --> K[select_top_k]
		K --> P[build_tool_args]
		P --> E{Single deterministic tool?}
		E -->|yes| T1[Direct tool execution]
		E -->|no| T2[LLM agent with shortlist]
		T2 --> F{Agent failure/timeout?}
		F -->|yes| T3[Direct tool fallback + summarization]
		F -->|no| T4[Agent final answer]

		D1 --> OUT[Response payload + metrics]
		T1 --> OUT
		T3 --> OUT
		T4 --> OUT
```

Optimization intent:

- reduce tool context surface
- reduce unnecessary tool calls
- keep direct/low-latency path for simple tasks

---

## 6) MCP server tools

### Search

- `search_articles`
- `search_research_papers`
- `search_lit_reviews`

### Translation

- `translate_japanese`
- `translate_french`
- `translate_spanish`

### Paraphrasing

- `paraphrase_formal`
- `paraphrase_casual`
- `paraphrase_academic`

---

## 7) Metrics and persistence

Each query run writes one row into `runs.db`.

Tracked fields include:

- `mode`
- `latency_ms`
- `tools_considered`
- `selected_tools_count`
- `tool_calls_executed`
- `router_confidence`
- `input_tokens`, `output_tokens` (best effort)
- `estimated_cost_usd`
- `success`
- `error_type`
- `selected_tools_json`
- `metadata_json`

### Metrics pipeline

```mermaid
flowchart LR
		R[Runtime response] --> X[extract_token_usage]
		X --> C[estimate_cost_usd]
		C --> M[QueryMetrics dataclass]
		M --> S[MetricsStore.write]
		S --> DB[(runs.db)]
```

---

## 8) Comparison math

For `compare` mode and benchmarking:

$$
	ext{latency\_improvement\_pct} = \frac{L_{baseline} - L_{optimized}}{L_{baseline}} \times 100
$$

$$
	ext{tools\_reduction\_pct} = \frac{T_{baseline} - T_{optimized}}{T_{baseline}} \times 100
$$

$$
	ext{cost\_improvement\_pct} = \frac{C_{baseline} - C_{optimized}}{C_{baseline}} \times 100
$$

---

## 9) Streamlit UI behavior

The dashboard in `app.py` provides:

1. **Live Comparison**  
	 Baseline on top, Optimized below, with prominent metrics and status.

2. **Benchmark Summary**  
	 Aggregate metrics across `eval_queries.jsonl`.

3. **Run History**  
	 Last runs from `runs.db` in tabular form.

UI flow:

```mermaid
flowchart TD
		U[User clicks Run baseline vs optimized] --> C[compare_query_modes]
		C --> B[baseline handle_user_query]
		C --> O[optimized handle_user_query]
		B --> DB[(runs.db)]
		O --> DB
		C --> V[Render comparison cards + answers]
		DB --> H[Render run history table]
```

---

## 10) Configuration (.env)

Create `.env` in project root:

```env
GROQ_API_KEY=your_groq_api_key_here

# optional; defaults to ./server.py
MCP_SERVER_PATH=C:/Users/lenovo/Desktop/MCP/MCP-Optimization/server.py

# reliability/performance controls
AGENT_TIMEOUT_SECONDS=60
TOOL_TIMEOUT_SECONDS=20

# arXiv lookup switch in server search path
USE_ARXIV=0
```

---

## 11) Setup and run

Install:

```bash
pip install -r requirements.txt
```

Run optimized query:

```bash
python client.py --mode optimized --query "Find recent research papers on MCP optimization"
```

Run baseline query:

```bash
python client.py --mode baseline --query "Find recent research papers on MCP optimization"
```

Run side-by-side compare:

```bash
python client.py --mode compare --query "Translate this to Japanese: Thank you for your support"
```

Run benchmark:

```bash
python benchmark.py
```

Launch dashboard:

```bash
streamlit run app.py
```

---

## 12) Reliability behavior and fallbacks

The orchestrator is designed to return useful output even during partial failures:

- router parse failure → fallback plan
- agent timeout/failure → direct tool fallback + summary
- outer exception → direct model fallback answer
- missing optional dependencies in server tools → graceful fallback messages

---

## 13) Current limitations

- Token usage extraction depends on provider response shape (best effort).
- Translation backend can vary by `googletrans` version/network behavior.
- Search quality/latency can vary by external API response times.
- Comparison results can be query-dependent (optimized may occasionally call more tools).

---

## 14) Recommended next improvements

- Add strict schema validation for router-selected `tool_args`.
- Add per-tool cooldown and historical success weighting.
- Add richer evaluation set with correctness labels.
- Add charts in UI for trend lines over time.