# Gemini Playground 🤖🚀

*"Gemini, by someone who reads the Gemini docs religiously."*

The **Gemini Playground** is a production-quality reference implementation of a Gemini-native AI platform. Built with FastAPI and LangGraph, it functions as a **simulation**: *what would a Google developer working on Gemini capabilities ship?*

This project demonstrates hands-on expertise in LLM platform engineering, specifically focusing on the unique, high-impact capabilities of the Google Gemini ecosystem.

---

## 🌟 Gemini-Native Features

Unlike generic multi-provider wrappers, this playground leans deep into Gemini-distinctive capabilities:

*   **Live API (Real-time Voice/Video):** Bidirectional WebSocket sessions for low-latency multimodal interaction.
*   **Multimodal Files API:** Native support for image, audio, video, and PDF grounding in a single request.
*   **Native Tools:** Grounded Google Search with citations, and sandboxed code execution.
*   **Imagen Generation:** High-quality text-to-image generation integrated directly into the API surface.
*   **FastMCP Server:** Every registered tool is automatically exposed to external MCP clients (Claude Desktop, Cursor).

---

## 🏗️ Architecture

The platform is designed to be **GCP-native** and **stateless**, scaling seamlessly on Cloud Run.

```
                       ┌──────────────────────────────────────────────────────────┐
                       │              Cloud Run — ai-platform                     │
                       │          (Scales to zero = $0 idle cost)                 │
                       │                                                          │
  Client               │  /api/v1/                                                │
  (SDK / UI)  ────────▶│    auth/          → API key management                  │
                       │    agents/        → single-agent (Gemini)                │
                       │    threads/       → conversation history                 │
                       │    files/         → multimodal uploads (Files API)       │
                       │    live/          → real-time voice/video (WebSocket)    │
                       │    evals/         → dataset + grader runs                │
                       │  /mcp/sse         → FastMCP server (all tools exposed)   │
                       │                                                          │
                       └────────┬─────────────────────────────────────────────────┘
                                │  ADC (service account — no API keys in prod)
          ┌─────────────────────┼─────────────────────────────────────────────┐
          │                     │                                             │
          ▼                     ▼                                             ▼
   PostgresDB (Neon)       Gemini / Vertex AI                       Cloud Operations
   + VectorDB (Pinecone)   - Gemini Flash 2.x (primary)             Cloud Logging
                           - Imagen 4.0 (generation)                Cloud Trace
                           - Native search grounding                Secret Manager
```

---

## 🛠️ Stack

*   **Backend:** FastAPI (Python 3.11)
*   **Orchestration:** LangGraph + LangChain
*   **SDK:** `google-genai` (Official Google Gemini SDK)
*   **Database:** SQLAlchemy + Postgres (Neon)
*   **Vector Store:** Pinecone
*   **Infrastructure:** GCP (Cloud Run, Secret Manager, Cloud Trace, Cloud Logging)
*   **Tools:** FastMCP (MCP Server)
