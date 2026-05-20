from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types

from app.config import settings

logger = logging.getLogger(__name__)
client = genai.Client(api_key=settings.gemini_api_key)


async def live_session_handler(
    websocket: WebSocket, model: str, config_dict: dict[str, Any] | None = None
) -> None:
    """
    Bridge between FastAPI WebSocket and Gemini Live API.
    Handles bidirectional streaming of audio/video and tool calls.
    """
    try:
        # Build live session config
        # Note: In a real implementation, we'd extract tools from the config_dict or agent preset.
        # For now, we'll support a basic config with native search if requested.

        tools = []
        if config_dict and "native_tools" in config_dict:
            nt = config_dict["native_tools"]
            if "search" in nt:
                tools.append(types.Tool(google_search=types.GoogleSearch()))
            if "code" in nt:
                tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

        live_config = types.LiveConnectConfig(
            tools=tools if tools else None,
            generation_config=types.GenerationConfig(
                response_modalities=["AUDIO"],
            ),
        )

        async with client.aio.live.connect(model=model, config=live_config) as session:
            logger.info(f"Connected to Gemini Live session with model: {model}")

            # Two concurrent tasks:
            # 1. Forward WS messages to Gemini
            # 2. Forward Gemini responses to WS

            async def send_to_gemini():
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)

                        # Handle different message types from client
                        msg_type = data.get("type")
                        if msg_type == "text":
                            await session.send(input=data.get("content"), end_of_turn=True)
                        elif msg_type == "audio":
                            audio_bytes = base64.b64decode(data.get("data"))
                            await session.send(
                                input=types.Part.from_bytes(data=audio_bytes, mime_type="audio/pcm")
                            )
                        elif msg_type == "video":
                            video_bytes = base64.b64decode(data.get("data"))
                            await session.send(
                                input=types.Part.from_bytes(
                                    data=video_bytes, mime_type="image/jpeg"
                                )
                            )
                        elif msg_type == "tool_response":
                            # Forward tool outputs back to Gemini
                            await session.send(
                                types.LiveClientToolResponse(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=data.get("name"),
                                            id=data.get("id"),
                                            response=data.get("response"),
                                        )
                                    ]
                                )
                            )
                except WebSocketDisconnect:
                    logger.info("WebSocket disconnected (client side)")
                except Exception as e:
                    logger.error(f"Error sending to Gemini: {e}")

            async def receive_from_gemini():
                try:
                    async for message in session.receive():
                        # message is a types.LiveServerMessage
                        response_data: dict[str, Any] = {}

                        if message.server_content:
                            content = message.server_content.model_turn
                            if content:
                                for part in content.parts:
                                    if part.text:
                                        response_data = {"type": "text", "content": part.text}
                                        await websocket.send_text(json.dumps(response_data))
                                    if part.inline_data:
                                        # Audio output
                                        audio_b64 = base64.b64encode(part.inline_data.data).decode(
                                            "utf-8"
                                        )
                                        response_data = {
                                            "type": "audio",
                                            "data": audio_b64,
                                            "mime_type": part.inline_data.mime_type,
                                        }
                                        await websocket.send_text(json.dumps(response_data))

                        if message.tool_call:
                            # Forward tool calls to client
                            for call in message.tool_call.function_calls:
                                response_data = {
                                    "type": "tool_call",
                                    "name": call.name,
                                    "id": call.id,
                                    "args": call.args,
                                }
                                await websocket.send_text(json.dumps(response_data))

                except Exception as e:
                    logger.error(f"Error receiving from Gemini: {e}")

            # Run both loops
            await asyncio.gather(send_to_gemini(), receive_from_gemini())

    except Exception as e:
        logger.error(f"Live session error: {e}")
        try:
            await websocket.close(code=1011, reason="Live session error")
        except Exception:
            pass
