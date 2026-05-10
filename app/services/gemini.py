from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Any

from google import genai
from google.genai import types

from app.config import settings
from app.tools import call_tool, get_tools

logger = logging.getLogger(__name__)
client = genai.Client(api_key=settings.gemini_api_key)


def list_gemini_models() -> list[str]:
    try:
        logger.info("Fetching Gemini models...")
        models = sorted(client.models.list(), key=lambda m: m.name or "")
        model_list: list[str] = []
        for m in models:
            if m.name:
                response = m.name.replace("models/", "")
                model_list.append(response)
    
        return model_list
    
    except Exception as e:
        logger.error(f"Error fetching Gemini models: {e}")
        raise


def gemini_service(model: str, prompt: str) -> str:
    try:
        logger.info(f"Generating content with Gemini model: {model}")
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction="You are a helpful assistant.",
                temperature=0.2,
                # thinking_config=types.ThinkingConfig(
                #     thinking_level="high"
                # ),
            )
        )

        return response.text or ""

    except Exception as e:
        logger.error(f"Error generating Gemini content: {e}")
        raise
    

def tools_service(model: str, prompt: str) -> str:
    """
    Generate content using Gemini with tool calling support using ChatSession.
    """
    try:
        logger.info(f"Starting tool-enabled chat with Gemini model: {model}")
        
        # Prepare tools from the registry
        raw_schemas = get_tools()
        tool_schemas: list[dict[str, Any]] = raw_schemas if isinstance(raw_schemas, list) else []
        gemini_tools: list[types.Tool] = []
        if tool_schemas:
            function_declarations = [
                types.FunctionDeclaration(
                    name=ts['function']['name'],
                    description=ts['function']['description'],
                    parameters=ts['function']['parameters']
                ) for ts in tool_schemas
            ]
            gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Initialize ChatSession (handles history automatically)
        chat = client.chats.create(
            model=model,
            config=types.GenerateContentConfig(
                tools=gemini_tools, # type: ignore[arg-type]
                system_instruction="You are a helpful assistant that uses tools when necessary.",
                temperature=0.1,
            )
        )

        # Tool Loop: Handle multi-turn tool calls robustly
        response = chat.send_message(prompt)
        for _ in range(10):
            # Extract all tool calls from the current response
            tool_calls = []
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.function_call:
                                tool_calls.append(part.function_call)
            
            if not tool_calls:
                return response.text or ""

            # Execute all tool calls in parallel (sequentially in this loop)
            tool_responses: list[types.Part] = []
            for call in tool_calls:
                logger.info("Executing tool: %s", call.name)
                try:
                    # Map result to a FunctionResponse Part
                    args = call.args or {}
                    result = call_tool(call.name or "", **args)
                    tool_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=call.name,
                                response={'result': result}
                            )
                        )
                    )
                except Exception as e:
                    logger.error(f"Tool execution failed: {call.name} -> {e}")
                    tool_responses.append(
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=call.name,
                                response={'error': str(e)}
                            )
                        )
                    )
            response = chat.send_message(tool_responses)
        return "Error: Maximum tool-calling iterations reached."

    except Exception as e:
        logger.error(f"Error in tools_service: {e}")
        raise


async def gemini_stream_service(model: str, prompt: str) -> AsyncGenerator[str, None]:
    try:
        logger.info(f"Starting Gemini streaming generation with model: {model}")
        async with client.aio as async_client:
            response = await async_client.models.generate_content_stream(
                model=model,
                contents=prompt
            )
            async for chunk in response:
                if chunk.text:
                    yield chunk.text
    except Exception as e:
        logger.error(f"Error in Gemini streaming service: {e}")
        raise