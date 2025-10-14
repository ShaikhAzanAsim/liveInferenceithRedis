from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
from pathlib import Path
import asyncio

from app.config import MAX_UPLOAD_SIZE, ALLOWED_EXT
from app.utils.video import save_upload_to_disk
from app.tasks import run_inference_job
from app.utils.redis_client import get_redis

router = APIRouter()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...), model: str = Form(...)):
    # validate content length (UploadFile doesn't give size directly, but SpooledTemporaryFile may)
    # Read file in chunks, saving to disk to avoid memory blow
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported extension {suffix}")

    # stream to disk
    dest = Path("uploads")  # using config.UPLOAD_DIR
    dest.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    temp_name = f"{job_id}{suffix}"
    temp_path = dest / temp_name
    size = 0
    with open(temp_path, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                out_f.close()
                temp_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File too large")
            out_f.write(chunk)

    # spawn background job
    loop = asyncio.get_event_loop()
    # create a task but do not await - let it run in background
    loop.create_task(run_inference_job(job_id, temp_path, model))

    ws_path = f"/ws/jobs/{job_id}"
    return JSONResponse({"job_id": job_id, "ws": ws_path})
