from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from google.genai import types

from app.config import build_genai_client
from app.services.gemini import build_native_tools

logger = logging.getLogger(__name__)
client = build_genai_client()


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

        nt: list[str] = (config_dict.get("native_tools") or []) if config_dict else []
        tools = build_native_tools(
            grounding="search" in nt,
            code_exec="code" in nt,
            url_context="url" in nt,
            location="location" in nt,
        )

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

            async def send_to_gemini() -> None:
                try:
                    while True:
                        message = await websocket.receive_text()
                        data = json.loads(message)

                        # Handle different message types from client
                        msg_type = data.get("type")
                        if msg_type == "text":
                            await session.send(input=data.get("content"), end_of_turn=True)
                        elif msg_type == "audio":
                            audio_content = data.get("data")
                            if audio_content:
                                audio_bytes = base64.b64decode(audio_content)
                                await session.send(
                                    input=types.Part.from_bytes(
                                        data=audio_bytes, mime_type="audio/pcm"
                                    )
                                )
                        elif msg_type == "video":
                            video_content = data.get("data")
                            if video_content:
                                video_bytes = base64.b64decode(video_content)
                                await session.send(
                                    input=types.Part.from_bytes(
                                        data=video_bytes, mime_type="image/jpeg"
                                    )
                                )
                        elif msg_type == "tool_response":
                            # Forward tool outputs back to Gemini
                            await session.send(
                                input=types.LiveClientToolResponse(
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

            async def receive_from_gemini() -> None:
                try:
                    async for message in session.receive():
                        # message is a types.LiveServerMessage
                        response_data: dict[str, Any] = {}

                        if message.server_content:
                            content = message.server_content.model_turn
                            if content and content.parts:
                                for part in content.parts:
                                    if part.text:
                                        response_data = {"type": "text", "content": part.text}
                                        await websocket.send_text(json.dumps(response_data))
                                    if part.inline_data and part.inline_data.data:
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

                        if message.tool_call and message.tool_call.function_calls:
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

            # Run both loops; cancel the sibling when either exits so the
            # Live session context manager is properly torn down.
            send_task = asyncio.create_task(send_to_gemini())
            recv_task = asyncio.create_task(receive_from_gemini())
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    except Exception as e:
        logger.error(f"Live session error: {e}")
        try:
            await websocket.close(code=1011, reason="Live session error")
        except Exception:
            pass
