import asyncio
import base64
import json
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from app.utils.redis_client import get_redis
from app.utils.video import read_video_metadata
from app.utils.metrics import now_ts, compute_metrics
from app.config import REDIS_TTL_SECONDS, MODEL_MAP

# In-memory WebSocket registry for active clients
WS_REGISTRY: Dict[str, set] = {}


# ============================================================
# 🟢 WebSocket Management
# ============================================================
def register_ws(job_id: str, ws):
    if job_id not in WS_REGISTRY:
        WS_REGISTRY[job_id] = set()
    WS_REGISTRY[job_id].add(ws)


def unregister_ws(job_id: str, ws):
    if job_id in WS_REGISTRY:
        WS_REGISTRY[job_id].discard(ws)
        if not WS_REGISTRY[job_id]:
            del WS_REGISTRY[job_id]


async def send_ws_message(job_id: str, message: Dict[str, Any]):
    conns = list(WS_REGISTRY.get(job_id, []))
    if not conns:
        return
    msg = json.dumps(message)
    for ws in conns:
        try:
            await ws.send_text(msg)
        except Exception:
            pass


async def send_ws_frame_b64(job_id: str, frame_b64: str, frame_idx: int):
    conns = list(WS_REGISTRY.get(job_id, []))
    if not conns:
        return
    payload = json.dumps({"type": "frame", "frame": frame_idx, "data": frame_b64})
    for ws in conns:
        try:
            await ws.send_text(payload)
        except Exception:
            pass


# ============================================================
# 🧠 Inference Job
# ============================================================
def normalize_model_key(model_path: str) -> str:
    """Return consistent model key without UUID prefix."""
    return Path(model_path).name.split("_", 1)[-1] if "_" in Path(model_path).name else Path(model_path).name


async def run_inference_job(job_id: str, video_path: Path, model_name: str):
    redis = get_redis()
    meta_key = f"job:{job_id}:meta"
    frames_key = f"job:{job_id}:frames"

    await redis.hset(meta_key, mapping={"model": model_name, "status": "running"})
    await redis.expire(meta_key, REDIS_TTL_SECONDS)
    await redis.expire(frames_key, REDIS_TTL_SECONDS)

    try:
        meta = read_video_metadata(video_path)
        total_frames = meta.get("total_frames") or 0
        fps = meta.get("fps") or 30.0
    except Exception as exc:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(exc)})
        await send_ws_message(job_id, {"type": "error", "message": str(exc)})
        return

    # ============================================================
    # 🎨 Load Custom Class Colors (using normalized model key)
    # ============================================================
    normalized_key = normalize_model_key(model_name)
    colors_key = f"model:{normalized_key}:colors"
    try:
        if await redis.exists(colors_key):
            color_data = await redis.get(colors_key)
            custom_colors = json.loads(color_data)
            print(f"✅ Loaded custom colors for {normalized_key}: {custom_colors}")
        else:
            custom_colors = {}
            print(f"🎨 No saved colors for {normalized_key}, using default palette.")
    except Exception as e:
        print(f"⚠️ Failed to load custom colors: {e}")
        custom_colors = {}

    # ============================================================
    # 🚀 Load ModelRunner with colors
    # ============================================================
    from app.models.runner import ModelRunner
    model_spec = MODEL_MAP.get(model_name, model_name)
    try:
        runner = ModelRunner(model_spec)
        runner.class_colors = custom_colors
    except Exception as e:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(e)})
        await send_ws_message(job_id, {"type": "error", "message": f"Model load failed: {e}"})
        return

    # ============================================================
    # 🎥 Process Video
    # ============================================================
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await redis.hset(meta_key, mapping={"status": "failed", "error": "Cannot open video"})
        await send_ws_message(job_id, {"type": "error", "message": "Cannot open video"})
        return

    start_ts = now_ts()
    t_pre, t_inf, t_post = [], [], []
    processed = 0
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = now_ts()
            try:
                res = runner.predict(frame)
            except Exception as e:
                print(f"⚠️ Inference failed on frame {frame_idx}: {e}")
                res = {"result_frame": frame}
            t_inf.append(now_ts() - t0)

            out_frame = res.get("result_frame", frame)
            success, jpg = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                frame_idx += 1
                continue

            jpg_bytes = jpg.tobytes()
            await redis.rpush(frames_key, jpg_bytes)
            await redis.expire(frames_key, REDIS_TTL_SECONDS)
            await redis.hset(meta_key, mapping={"processed_frames": frame_idx + 1, "total_frames": total_frames})
            await redis.expire(meta_key, REDIS_TTL_SECONDS)

            b64 = base64.b64encode(jpg_bytes).decode("ascii")
            await send_ws_frame_b64(job_id, b64, frame_idx)

            pct = ((frame_idx + 1) / total_frames * 100.0) if total_frames else 0.0
            await send_ws_message(
                job_id,
                {"type": "progress", "frame": frame_idx + 1, "total_frames": total_frames, "pct": round(pct, 2)},
            )

            frame_idx += 1
            processed += 1
            await asyncio.sleep(0)

    except Exception as e:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(e)})
        await send_ws_message(job_id, {"type": "error", "message": str(e)})

    finally:
        cap.release()

    end_ts = now_ts()
    metrics = compute_metrics(t_pre, t_inf, t_post, start_ts, end_ts, processed, model_name)

    await redis.hset(
        meta_key,
        mapping={
            "status": "done",
            "processed_frames": processed,
            "total_frames": total_frames,
            "start_ts": start_ts,
            "end_ts": end_ts,
            **{k: str(v) for k, v in metrics.items()},
        },
    )
    await redis.expire(meta_key, REDIS_TTL_SECONDS)
    await redis.expire(frames_key, REDIS_TTL_SECONDS)
    await send_ws_message(job_id, {"type": "done", "metrics": metrics})
