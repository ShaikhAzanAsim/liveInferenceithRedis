from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uuid
from pathlib import Path
import asyncio
import json
import os

from app.utils.redis_client import get_redis
from app.config import MAX_UPLOAD_SIZE, ALLOWED_EXT
from app.tasks import run_inference_job
from app.models.runner import ModelRunner

router = APIRouter()

# 🧠 Store reference to currently active model
active_model_runner: ModelRunner | None = None
active_model_name: str | None = None  # ✅ Track the model name/path (normalized)


def normalize_model_key(model_path: str) -> str:
    """
    Normalize model path to a consistent Redis key (ignores UUID prefix).
    Example: custom_models/00abcd_bag.pt → bag.pt
    """
    return Path(model_path).name.split("_", 1)[-1] if "_" in Path(model_path).name else Path(model_path).name


@router.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    model: str = Form("yolov8n"),
    custom_model: UploadFile = File(None)
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported video extension {suffix}")

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

    model_path = None
    global active_model_runner, active_model_name

    # Handle custom model upload
    if custom_model:
        if not custom_model.filename.endswith(".pt"):
            raise HTTPException(status_code=400, detail="Only .pt model files are allowed")

        model_dir = Path("custom_models")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{job_id}_{custom_model.filename}"

        with open(model_path, "wb") as mf:
            while chunk := await custom_model.read(1024 * 1024):
                mf.write(chunk)

        model = str(model_path)

        # ✅ Initialize ModelRunner
        active_model_runner = ModelRunner(model)
        active_model_name = normalize_model_key(model)
        print(f"✅ Active custom model loaded: {model}")

    else:
        # ✅ Handle built-in model
        active_model_runner = ModelRunner(model)
        active_model_name = model
        print(f"✅ Active default model set: {model}")

    # Start inference job
    loop = asyncio.get_event_loop()
    loop.create_task(run_inference_job(job_id, video_path, model))

    return JSONResponse({
        "job_id": job_id,
        "ws": f"/ws/jobs/{job_id}",
        "model_used": model
    })


# 🎨 === Save Class Colors ===
@router.post("/set_class_colors")
async def set_class_colors(request: Request):
    """
    Save custom colors for a specific model in Redis.
    Frontend must send JSON:
    { "model_name": "bag.pt", "colors": { "class1": "#FF0000", ... } }
    """
    try:
        data = await request.json()
        colors = data.get("colors")
        model_name = data.get("model_name")

        if not model_name:
            raise HTTPException(status_code=400, detail="Missing 'model_name' field.")
        if not isinstance(colors, dict):
            raise HTTPException(status_code=400, detail="Invalid color data format")

        # ✅ Normalize model key (ignore UUID prefix)
        def normalize_model_key(model_path: str) -> str:
            return Path(model_path).name.split("_", 1)[-1] if "_" in Path(model_path).name else Path(model_path).name

        normalized_key = normalize_model_key(model_name)

        redis = get_redis()
        colors_key = f"model:{normalized_key}:colors"
        await redis.set(colors_key, json.dumps(colors))

        print(f"🎨 Colors saved for {normalized_key}: {colors}")
        return JSONResponse({"message": f"Colors saved for {normalized_key}"})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

