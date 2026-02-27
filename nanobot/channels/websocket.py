"""WebSocket channel implementation for browser-based chat interface."""

import asyncio
import json
from typing import Any

import websockets
from loguru import logger
from websockets.server import WebSocketServerProtocol

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import BaseModel



class WebSocketChannel(BaseChannel):
    """WebSocket channel for browser-based chat interface."""
    
    name = "websocket"
    
    def __init__(self, config: WebSocketConfig, bus: MessageBus):
        super().__init__(config, bus)
        self.config: WebSocketConfig = config
        self._server: Any = None
        self._connections: set[WebSocketServerProtocol] = set()
        self._session_map: dict[WebSocketServerProtocol, str] = {}
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        if not self.config.enabled:
            logger.warning("WebSocket channel is disabled in config")
            return
        
        self._running = True
        
        try:
            # Handler for websockets v13+ (only receives websocket, not path)
            async def connection_handler(websocket):
                await self._handle_connection(websocket)
            
            async with websockets.serve(
                connection_handler,
                self.config.host,
                self.config.port,
                ping_interval=20,   # Ping every 20 seconds
                ping_timeout=120,   # Wait up to 2 minutes for pong (for slow LLM)
                close_timeout=10
            ):
                logger.info(
                    f"✅ WebSocket channel listening on "
                    f"ws://{self.config.host}:{self.config.port}"
                )
                # Keep server running
                while self._running:
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        self._running = False
        
        # Close all active connections
        close_tasks = [ws.close() for ws in self._connections]
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        self._connections.clear()
        self._session_map.clear()
        
        logger.info("WebSocket channel stopped")
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message to a specific WebSocket client."""
        # Find the WebSocket connection for this chat_id
        target_ws = None
        for ws, session_id in self._session_map.items():
            if session_id == msg.chat_id:
                target_ws = ws
                break
        
        if not target_ws or target_ws not in self._connections:
            logger.warning(f"No active WebSocket connection for chat_id: {msg.chat_id}")
            return
        
        try:
            await target_ws.send(json.dumps({
                "type": "agent_response",
                "content": msg.content,
                "timestamp": msg.metadata.get("timestamp", ""),
                "session_id": msg.chat_id,
                "reply_to": msg.reply_to,
                "metadata": msg.metadata
            }))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def _handle_connection(
        self,
        websocket: WebSocketServerProtocol
    ) -> None:
        """Handle a new WebSocket connection."""
        client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
        
        # Check IP allowlist
        if self.config.allow_from and client_ip not in self.config.allow_from:
            logger.warning(
                f"WebSocket connection rejected from {client_ip} "
                f"(not in allowFrom list)"
            )
            await websocket.close(1008, "Unauthorized")
            return
        
        # Generate session ID
        session_id = f"{client_ip}_{id(websocket)}"
        
        logger.info(f"New WebSocket connection from {client_ip} (session: {session_id})")
        
        self._connections.add(websocket)
        self._session_map[websocket] = session_id
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "status",
                "content": "Connected to nanobot",
                "timestamp": "",
                "session_id": session_id
            }))
            
            # Message handling loop
            async for raw_message in websocket:
                await self._handle_incoming_message(
                    websocket,
                    raw_message,
                    session_id,
                    client_ip
                )
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {client_ip}")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}", exc_info=True)
        finally:
            self._connections.discard(websocket)
            self._session_map.pop(websocket, None)
    
    async def _handle_incoming_message(
        self,
        websocket: WebSocketServerProtocol,
        raw_message: str,
        session_id: str,
        client_ip: str
    ) -> None:
        """Process an incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            message_type = data.get("type")
            
            if message_type == "user_message":
                content = data.get("content", "")
                if not content:
                    return
                
                logger.info(f"WebSocket message from {client_ip}: {content[:100]}")
                
                # CRITICAL: Use background task to prevent blocking the WebSocket connection
                # This allows the WebSocket to continue receiving pings while the agent
                # processes the message (which might take a long time if calling LLM)
                asyncio.create_task(
                    self._handle_message(
                        sender_id=client_ip,
                        chat_id=session_id,
                        content=content,
                        media=data.get("media", []),
                        metadata={
                            "websocket_id": id(websocket),
                            "client_ip": client_ip,
                            **data.get("metadata", {})
                        }
                    )
                )
                
            elif message_type == "ping":
                # Respond to heartbeat ping
                await websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": "",
                    "session_id": session_id
                }))
                
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON from WebSocket: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "content": "Invalid JSON format",
                "timestamp": "",
                "session_id": session_id
            }))
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}", exc_info=True)
            await websocket.send(json.dumps({
                "type": "error",
                "content": str(e),
                "timestamp": "",
                "session_id": session_id
            }))