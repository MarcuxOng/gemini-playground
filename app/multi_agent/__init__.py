"""
Gemini-native multi-agent systems. Coordination value must come from Gemini's
own capabilities (Live API, caching, multimodal transport, A2A), not hardcoded routing.
"""

from app.multi_agent.a2a import A2ARouter, AgentCard, build_agent_card
from app.multi_agent.protocol import AgentMessage, AgentPart, agent_message_to_gemini_parts
from app.multi_agent.shared_context import SharedContext

__all__ = [
    "A2ARouter",
    "AgentCard",
    "SharedContext",
    "AgentPart",
    "AgentMessage",
    "agent_message_to_gemini_parts",
    "build_agent_card",
]
