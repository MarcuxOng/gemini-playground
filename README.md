# Gemini Playground

*"Gemini, by someone who reads the Gemini docs religiously."*

The **Gemini Playground** is a production-quality reference implementation of a Gemini-native AI platform. Built with FastAPI and LangGraph, it is a **Gemini capability layer** — the shared backend for three product candidates (A: corpus core, C: measurement lab, D: second-opinion engine), documented under `.misc/api/docs/ideas/`.

---

## Purpose

Every Gemini call from every product routes through this one backend: MCP at `/mcp/sse` (16 registered tools) and HTTP at `/api/v1/`. The platform is **Gemini-only** (no provider abstraction), GCP-native, and the one chokepoint for paid Gemini calls.

The platform is:
- A learning sandbox for exploring LLM platform engineering, Gemini-native APIs, GCP infrastructure.
- A portfolio piece — production-quality code and demos that can be linked from a CV.
- A Gemini-native reference — no multi-provider parity.

It is **not** a SaaS, an open-source library, or a Claude/GPT clone.

---

## Features

### Core Gemini
- **Text Generation** — streaming and non-streaming, with native search grounding, code execution, and URL context tools
- **Structured Output** — strict JSON schemas, guaranteed shape
- **Multimodal Files API** — PDF, image, audio, video, with native understanding (no conversion step)
- **Context Caching** — cache large documents once on Google's side and reuse across requests
- **Native Search Grounding** — real web citations, with source links appended
- **Maps Grounding** — place names, addresses, and distances with a **Places:** citation block

### Agents
- **ReAct Agents** — LangGraph `create_react_agent` presets: coder, research, analyst, knowledge
- **Consensus Engine** — parallel Flash ensemble + Pro-tier judge synthesis
- **Shared Context** — cache-backed shared context across agents
- **Multimodal Inter-Agent Protocol** — pass raw Parts between agents

### Developer Experience
- **MCP Server** — every registered tool auto-exposed at `/mcp/sse`
- **HTTP API** — 11 routers, with `x-api-key` auth and per-route rate limiting
- **Eval Harness** — Gemini-as-judge evals, with dataset and run management
- **Observability** — OpenTelemetry → Cloud Trace, structured JSON → Cloud Logging

---

## Architecture

The platform is **GCP-native** and **stateless**, scaling seamlessly on Cloud Run.

```text
                        ┌──────────────────────────────────────────────────────────┐
                        │                Cloud Run — ai-platform                   │
                        │                                                          │
  Client                │  /api/v1/                                                │
  (React SPA) ────────▶│    auth/          → API key management                   │
                        │    agents/        → ReAct agents (LangGraph)             │
                        │    threads/       → conversation history                 │
                        │    rag/           → document ingest + query              │
                        │    files/         → multimodal uploads                   │
                        │    caches/        → context caching                      │
                        │    evals/         → eval harness                         │
                        │    mcp-servers/   → external MCP server admin            │
                        │    multi-agent/   → consensus engine + shared context    │
                        │    health/        → dependency probes                    │
                        │                                                          │
                        │  /mcp/sse       → FastMCP SSE server (16 tools)          │
                        └──────────────────────────────────────────────────────────┘
                                  │
                       ┌──────────┴──────────┐
                       │  Gemini / Vertex AI │
                       └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API framework | FastAPI + Uvicorn |
| LLM — default path | LangChain (`langchain-google-vertexai`, `langchain-google-genai`) |
| LLM — native path | Google Gen AI SDK (`google-genai`) |
| Agent framework | LangGraph (`create_react_agent`) |
| Database ORM | SQLAlchemy + Alembic-free auto-create |
| Database | Neon (Postgres, free tier) / SQLite for local |
| Vector store | Pinecone (raw SDK) |
| MCP server | FastMCP (SSE transport) |
| Rate limiting | SlowAPI + Redis with in-memory fallback |
| Observability | OpenTelemetry → Cloud Trace; structured JSON → Cloud Logging |
