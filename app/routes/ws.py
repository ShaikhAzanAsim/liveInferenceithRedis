from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.tasks import register_ws, unregister_ws
import json

router = APIRouter()

@router.websocket("/ws/jobs/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    register_ws(job_id, websocket)
    try:
        # send a welcome message
        await websocket.send_text(json.dumps({"type":"info","message":"connected","job_id":job_id}))
        while True:
            # keep connection alive; clients usually don't send messages, but we await for pings
            data = await websocket.receive_text()
            # optionally client can send {"action":"ping"} or {"action":"download"}
            # For now ignore or echo
            try:
                msg = json.loads(data)
            except Exception:
                msg = {}
            if msg.get("action") == "download":
                await websocket.send_text(json.dumps({"type":"info", "message":"Download requested; call /download/<job_id> to retrieve file"}))
    except WebSocketDisconnect:
        unregister_ws(job_id, websocket)
    except Exception:
        unregister_ws(job_id, websocket)
        try:
            await websocket.close()
        except Exception:
            pass
