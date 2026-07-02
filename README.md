# Gemini Playground

*"Gemini, by someone who reads the Gemini docs religiously."*

The **Gemini Playground** is a production-quality reference implementation of a Gemini-native AI platform. Built with FastAPI and LangGraph, it functions as a **simulation**: *what would a Google developer working on Gemini capabilities ship?*

A companion React SPA frontend (`playground-ui/`) provides a polished Google-design-language interface for every feature.

---

## Purpose

This project demonstrates hands-on expertise in LLM platform engineering, focusing on Gemini-distinctive capabilities rather than multi-provider parity. Every feature passes the filter: *"Would the Gemini team actually ship this?"*

The platform is:
- A learning sandbox for exploring LLM infrastructure
- A portfolio piece demonstrating production-quality code
- A Gemini-native reference — no Claude/GPT mimicry

It is **not** a SaaS, an open-source library, or a multi-provider playground.

---

## Features

### Core Gemini
- **Text Generation** — Streaming and non-streaming, with native search grounding, code execution, and URL context tools
- **Structured Output** — Schema-constrained JSON responses via Pydantic models
- **ReAct Agents** — LangGraph-powered agents with tool calling: Coder, Research, Analyst, Knowledge presets
- **Conversation Threads** — Persistent conversation history with LangGraph checkpointer

### Multimodal
- **Files API** — Upload and query images, audio, video, and PDFs in a single call
- **Live API** — Real-time bidirectional WebSocket voice/video sessions
- **Imagen** — Text-to-image generation and image editing

### Knowledge & Search
- **RAG Pipeline** — Gemini embeddings + Pinecone vector store, exposed as the `search_knowledge_base` tool
- **Native Search Grounding** — Gemini-grounded Google Search with citations

### Developer Tools
- **MCP Server** — Every registered tool auto-exposed at `/mcp/sse` for external clients (Claude Desktop, Cursor)
- **MCP Client** — Consume external MCP servers as agent tools
- **Evals Harness** — Gemini-as-judge grading with dataset management and run history
- **Context Caching** — Cache large contexts for repeated queries, with TTL management
- **API Keys** — Generate, list, and revoke API keys via `/api/v1/auth`

### Multi-Agent Orchestration
- **A2A Discovery** — Agent Cards at `/.well-known/agent.json` for peer discovery
- **Live API Swarm** — Interrupt injection into running agents via WebSocket
- **Consensus Engine** — Parallel Flash ensemble + Pro judge synthesis

---

## Architecture

The platform is **GCP-native** and **stateless**, scaling seamlessly on Cloud Run.

```text
                       ┌──────────────────────────────────────────────────────────┐
                       │              Cloud Run — ai-platform                     │
                       │          (Scales to zero = $0 idle cost)                 │
                       │                                                          │
  Client               │  /api/v1/                                                │
  (React SPA) ────────▶│    auth/          → API key management                  │
                       │    agents/        → ReAct agents (Gemini)                │
                       │    threads/       → conversation history                 │
                       │    rag/           → ingest + query (Pinecone)            │
                       │    files/         → multimodal uploads (Files API)       │
                       │    gemini/        → text + streaming + structured        │
                       │    image/         → generation + edit                    │
                       │    caches/        → context caching                      │
                       │    evals/         → datasets + grader runs               │
                       │    mcp-servers/   → external MCP server management       │
                       │    health/        → dependency status                    │
                       │  /api/v1/live/ws  → real-time voice/video (WebSocket)    │
                       │                                                          │
  MCP Clients          │  /mcp/sse         → FastMCP server (all tools exposed)   │
  (Claude/Cursor) ────▶│                                                          │
                       └────────┬─────────────────────────────────────────────────┘
                                │  ADC (service account — no API keys in prod)
          ┌─────────────────────┼─────────────────────────────────────────────┐
          │                     │                                             │
          ▼                     ▼                                             ▼
   Neon DB (Postgres)      Gemini / Vertex AI                        Cloud Operations
   Pinecone (Vectors)       - Gemini Flash 2.x (primary)             Cloud Logging ($0)
                            - Gemini Pro (complex tasks)             Cloud Trace ($0)
                            - Gemini Embedding 001                   Secret Manager ($0)
                            - Imagen 4.0 (generation)
                            - Live API (real-time)
                            - Native search grounding
```

---

## Tech Stack

- **FastAPI** (Python 3.11) — async REST API with OpenAPI docs
- **LangGraph** + **LangChain** — agent orchestration with checkpointed memory
- **google-genai** — official Gemini SDK (Live API, Imagen, Files, native tools)
- **SQLAlchemy** + **Postgres** (Neon, free tier) — relational data with `playground_v1_` prefix convention
- **Pinecone** (free tier) — vector store for RAG
- **FastMCP** — MCP server at `/mcp/sse` exposing all registered tools
- **ADC** — Cloud Run service account auth (no API keys in prod)
- **Redis** — rate limiting (proxied via limiter)
- **GCP** — Cloud Run, Secret Manager, Cloud Trace, Cloud Logging, Artifact Registry
- **GitHub Actions** — CI (pytest + ruff + mypy), CD (Cloud Build deploy)
