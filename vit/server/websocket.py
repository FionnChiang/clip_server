from fastapi import WebSocket
import asyncio


class WebSocketManager:
    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, job_id: str, ws: WebSocket):
        await ws.accept()
        if job_id not in self._connections:
            self._connections[job_id] = []
        self._connections[job_id].append(ws)

    def disconnect(self, job_id: str, ws: WebSocket):
        if job_id in self._connections:
            conns = self._connections[job_id]
            if ws in conns:
                conns.remove(ws)
            if not conns:
                del self._connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        if job_id not in self._connections:
            return
        dead = []
        for ws in self._connections[job_id]:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(job_id, ws)

    async def broadcast_to_all(self, message: dict):
        for job_id in list(self._connections.keys()):
            await self.broadcast(job_id, message)


ws_manager = WebSocketManager()
