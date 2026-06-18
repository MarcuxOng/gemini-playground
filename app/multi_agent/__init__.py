"""Gemini-native multi-agent systems. Coordination value must come from Gemini's
own capabilities (Live API, caching, multimodal transport, A2A), not hardcoded routing."""

from app.multi_agent.shared_context import SharedContext

__all__ = ["SharedContext"]
