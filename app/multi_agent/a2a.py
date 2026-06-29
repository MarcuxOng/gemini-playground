"""A2A discovery mesh — Agent Cards at /.well-known/agent.json + dynamic routing.

Implements the Agent-to-Agent (A2A) open standard: every hosted agent exposes an
Agent Card describing its capabilities, supported modalities, and active tools.
The A2ARouter discovers peer agents at runtime via ``httpx`` and routes tasks to
the best-suited one using Gemini capability scoring — no hardcoded routing table.

Reference: Google A2A Protocol
*https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/*
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app.config import default_model
from app.services.llm import build_llm
from app.tools import list_tool_names

logger = logging.getLogger(__name__)

# ── Preset metadata ────────────────────────────────────────────────────────────
# Descriptions and tool lists mirror the preset modules under app/agents/presets/.
# Kept inline to avoid circular imports from the preset package.

_PRESET_META: dict[str, dict[str, Any]] = {
    "research": {
        "description": "Research assistant with web search, weather, Wikipedia, and YouTube tools.",
        "tools": [
            "google_search",
            "get_weather",
            "get_datetime_info",
            "get_wikipedia_summary",
            "get_youtube_transcript",
        ],
    },
    "coder": {
        "description": "Senior software engineer with code generation, regex, file I/O, and math tools.",
        "tools": ["calculate", "read_file", "write_file", "test_regex", "count_tokens"],
    },
    "analyst": {
        "description": "Financial analyst with stock market and cryptocurrency data tools.",
        "tools": ["get_stock_price", "get_crypto_price"],
    },
    "knowledge": {
        "description": "Knowledge management assistant with private database RAG search.",
        "tools": ["search_knowledge_base", "calculate", "scrape_url"],
    },
}


# ── Pydantic models ────────────────────────────────────────────────────────────


class A2ACapability(BaseModel):
    """A single capability advertised by an agent in its Agent Card."""

    name: str
    description: str
    tools: list[str] = Field(default_factory=list)
    modalities: list[str] = Field(default_factory=lambda: ["text"])


class AgentCard(BaseModel):
    """A2A Agent Card — the JSON document exposed at ``/.well-known/agent.json``.

    Describes a hosted agent (or agent host) so that peers can discover its
    capabilities at runtime and decide whether to route tasks to it.
    """

    name: str
    description: str
    url: str
    version: str = "1.0"
    protocol: str = "a2a/1.0"
    capabilities: list[A2ACapability] = Field(default_factory=list)
    default_model: str = default_model
    mcp_tools: list[str] = Field(default_factory=list)
    invoke_url: str | None = None


# ── Agent Card builder ─────────────────────────────────────────────────────────


def build_agent_card(base_url: str, default_model: str = default_model) -> AgentCard:
    """Build the host server's Agent Card from known presets and the tool registry.

    Args:
        base_url: The public base URL of this server (e.g. ``http://localhost:8000``).
        default_model: Default Gemini model used by hosted agents.

    Returns:
        A fully populated :class:`AgentCard` ready for exposure at
        ``/.well-known/agent.json``.
    """
    url = base_url.rstrip("/")
    capabilities: list[A2ACapability] = []

    for preset_name, meta in _PRESET_META.items():
        capabilities.append(
            A2ACapability(
                name=preset_name,
                description=meta["description"],
                tools=meta["tools"],
            )
        )

    return AgentCard(
        name="Gemini Playground",
        description=(
            "A self-hosted AI/LLM platform exposing Gemini through a clean API surface "
            "with ReAct agents, RAG over Pinecone, multimodal Files API, Live API, "
            "Imagen generation, and an MCP server."
        ),
        url=url,
        capabilities=capabilities,
        default_model=default_model,
        mcp_tools=list_tool_names(),
        invoke_url=f"{url}/api/v1/agents/invoke",
    )


# ── A2A Router ─────────────────────────────────────────────────────────────────


_ROUTING_PROMPT = """\
You are an A2A routing agent. Given a task description, select the single best agent from the list below.

TASK: {task}

AGENTS:
{agent_list}

Respond with ONLY the numeric agent ID in brackets, e.g. [2]. No explanation, no punctuation."""


class PeerNotFoundError(Exception):
    """Raised when a peer agent``s Agent Card cannot be fetched or parsed."""

    def __init__(self, url: str, detail: str) -> None:
        self.url = url
        self.detail = detail
        super().__init__(f"Peer {url}: {detail}")


def _parse_routing_index(raw: str) -> int | None:
    """Extract a numeric index from a Geminid-generated bracket like ``[2]``."""
    import re

    match = re.search(r"\[(\d+)\]", raw)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


class A2ARouter:
    """Discovers peer agents and routes tasks to the best-suited one via Gemini.

    Usage::

        router = A2ARouter(host_card=build_agent_card("http://localhost:8000"))
        await router.discover(["https://peer1.example.com"])
        best_url, best_card = await router.route("What's the weather in Tokyo?")
        # → ("http://localhost:8000", <AgentCard name='Gemini Playground'>)
    """

    def __init__(
        self,
        host_card: AgentCard | None = None,
        timeout: float = 10.0,
    ) -> None:
        self._host = host_card
        self._timeout = timeout
        self._peers: dict[str, AgentCard] = {}

    # ── discovery ──────────────────────────────────────────────────────────

    async def discover(self, peer_urls: list[str]) -> list[str]:
        """Fetch Agent Cards from all *peer_urls* via ``GET /.well-known/agent.json``.

        Returns the subset of URLs that were successfully discovered and parsed.
        Failures are logged but do not prevent discovery of other peers.
        """
        discovered: list[str] = []
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            for url in peer_urls:
                card_url = f"{url.rstrip('/')}/.well-known/agent.json"
                try:
                    resp = await client.get(card_url)
                    resp.raise_for_status()
                    card = AgentCard.model_validate(resp.json())
                    self._peers[url] = card
                    discovered.append(url)
                    logger.info("Discovered peer at %s: %s", url, card.name)
                except httpx.HTTPError as exc:
                    logger.warning("Failed to fetch Agent Card from %s: %s", url, exc)
                except Exception as exc:
                    logger.warning("Invalid Agent Card from %s: %s", url, exc)
        return discovered

    @property
    def known_peers(self) -> dict[str, AgentCard]:
        """Return all successfully discovered peer Agent Cards."""
        return dict(self._peers)

    # ── routing ────────────────────────────────────────────────────────────

    async def route(self, task: str, model: str = default_model) -> tuple[str, AgentCard]:
        """Route *task* to the best-suited agent (host or peer) using Gemini.

        Builds a capability list from the host card (if provided) and all
        discovered peers, then asks Gemini to select the best agent for the
        given task description.

        Args:
            task: Natural-language task description.
            model: Gemini model used for capability scoring.

        Returns:
            Tuple of ``(peer_url_or_"host", AgentCard)``.

        Raises:
            ValueError: If no agents are available (no host card and no discovered peers).
        """
        candidates: list[tuple[str, AgentCard]] = []

        if self._host is not None:
            candidates.append(("host", self._host))
        for peer_url, peer_card in self._peers.items():
            candidates.append((peer_url, peer_card))

        if not candidates:
            raise ValueError(
                "No agents available for routing. Provide a host_card or discover peers."
            )

        if len(candidates) == 1:
            return candidates[0]

        # Assign a unique index to each candidate for unambiguous resolution
        index_map: dict[int, tuple[str, AgentCard]] = {}
        entries: list[str] = []
        idx = 0
        for source_url, card in candidates:
            for cap in card.capabilities:
                tools_str = ", ".join(cap.tools) if cap.tools else "none"
                source_label = "host" if source_url == "host" else source_url
                entries.append(
                    f"ID [{idx}]\n"
                    f"  Host: {source_label}\n"
                    f"  Description: {cap.description}\n"
                    f"  Tools: {tools_str}"
                )
                index_map[idx] = (source_url, card)
                idx += 1

        if not entries:
            raise ValueError(
                "No agent capabilities available for routing. "
                "Provide agents with at least one capability."
            )

        prompt = _ROUTING_PROMPT.format(
            task=task,
            agent_list="\n".join(entries),
        )

        llm = build_llm(model, temperature=0.0)
        result = await run_in_threadpool(llm.invoke, prompt)
        raw_selected = str(result.content).strip()

        parsed = _parse_routing_index(raw_selected)
        if parsed is not None and parsed in index_map:
            return index_map[parsed]

        raise ValueError(
            f"Gemini returned '{raw_selected}' but no candidate agent matches that identifier."
        )
