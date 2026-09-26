"""WebSocket fan-out: every connected dashboard gets every tick, event and status change."""
from __future__ import annotations

import asyncio
import logging

from fastapi import WebSocket

from ..store.db import to_json

log = logging.getLogger("reo.hub")


class Hub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, ws: WebSocket, first_message: dict) -> None:
        await ws.accept()
        self.clients.add(ws)
        await ws.send_text(to_json(first_message))

    def disconnect(self, ws: WebSocket) -> None:
        self.clients.discard(ws)

    async def broadcast(self, message: dict, timeout_s: float = 2.0) -> None:
        if not self.clients:
            return
        data = to_json(message)
        clients = list(self.clients)

        async def send(ws: WebSocket) -> bool:
            try:
                await asyncio.wait_for(ws.send_text(data), timeout_s)
                return True
            except Exception:  # gone or too slow: drop it so it can't stall the simulation
                return False

        results = await asyncio.gather(*(send(ws) for ws in clients))
        for ws, ok in zip(clients, results):
            if not ok:
                self.disconnect(ws)
