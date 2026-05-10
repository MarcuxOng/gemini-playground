"""
Converts external MCP server configs into LangChain-compatible tools
using langchain-mcp-adapters.
"""

from __future__ import annotations

import logging

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


async def load_mcp_tools(server_config: dict[str, object]) -> list[BaseTool]:
    """
    Connect to an external MCP server and return its tools as LangChain BaseTools.
    """
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
        transport = server_config.get("transport")
        
        # Infer transport if missing
        if not transport:
            if server_config.get("url"):
                transport = "sse"
            elif server_config.get("command"):
                transport = "stdio"
            else:
                raise ValueError(f"Could not infer transport for MCP server '{server_config.get('name')}'. Provide 'url' or 'command'.")

        if transport == "sse":
            servers = {
                server_config["name"]: {
                    "transport": "sse",
                    "url": server_config["url"],
                }
            }
        elif transport == "stdio":
            # SECURITY: Disabling stdio transport for user-provided configurations
            # to prevent arbitrary command execution.
            logger.error(f"Rejected stdio transport for MCP server '{server_config.get('name')}': stdio is disabled for user-registered servers.")
            raise ValueError("stdio transport is not allowed for user-registered servers due to security risks.")
        else:
            raise ValueError(f"Unsupported MCP transport: {transport}")

        async with MultiServerMCPClient(servers) as client:  # type: ignore[arg-type, misc]
            tools = await client.get_tools()
            logger.info(f"Loaded {len(tools)} tools from MCP server '{server_config['name']}'")
            return tools

    except ImportError:
        logger.error("langchain-mcp-adapters not installed. Run: pip install langchain-mcp-adapters")
        raise
    except Exception:
        logger.exception(f"Failed to load MCP tools from '{server_config.get('name')}'")
        raise