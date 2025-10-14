from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uuid
import os
from pathlib import Path
import asyncio

from app.config import MAX_UPLOAD_SIZE, ALLOWED_EXT
from app.tasks import run_inference_job

router = APIRouter()

@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    model: str = Form("yolov8n"),  # default model name
    custom_model: UploadFile = File(None)  # optional .pt model upload
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension {suffix}")

    # Save video
    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    video_path = upload_dir / f"{job_id}{suffix}"

    size = 0
    with open(video_path, "wb") as out_f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_SIZE:
                out_f.close()
                video_path.unlink(missing_ok=True)
                raise HTTPException(status_code=400, detail="File too large")
            out_f.write(chunk)

    # Handle custom YOLO model upload
    model_path = None
    if custom_model:
        if not custom_model.filename.endswith(".pt"):
            raise HTTPException(status_code=400, detail="Only .pt model files are allowed")

        model_dir = Path("custom_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{job_id}_{custom_model.filename}"

        with open(model_path, "wb") as mf:
            while chunk := await custom_model.read(1024 * 1024):
                mf.write(chunk)

        model = str(model_path)  # use uploaded model path instead of name

    # Spawn background job
    loop = asyncio.get_event_loop()
    loop.create_task(run_inference_job(job_id, video_path, model))

    return JSONResponse({
        "job_id": job_id,
        "ws": f"/ws/jobs/{job_id}",
        "model_used": model
    })
