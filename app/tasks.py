import asyncio
import base64
import json
import uuid
import os
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

from app.utils.redis_client import get_redis
from app.utils.video import read_video_metadata
from app.utils.metrics import now_ts, compute_metrics
from app.config import REDIS_TTL_SECONDS, MODEL_MAP

# in-memory WebSocket registry for active clients: job_id -> set of websocket objects
# NOTE: single-process only. For multi-worker use Redis Pub/Sub.
WS_REGISTRY: Dict[str, set] = {}

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
            # ignore broken websocket; cleanup will occur on disconnect
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

async def run_inference_job(job_id: str, video_path: Path, model_name: str):
    """
    Main background job to read frames, run inference, push frames to redis,
    and send progress + frame images to any listening WebSockets.
    """
    redis = get_redis()
    meta_key = f"job:{job_id}:meta"
    frames_key = f"job:{job_id}:frames"

    # store initial meta
    await redis.hset(meta_key, mapping={
        "model": model_name,
        "status": "running",
    })
    await redis.expire(meta_key, REDIS_TTL_SECONDS)
    await redis.expire(frames_key, REDIS_TTL_SECONDS)

    # Basic frame capture
    try:
        meta = read_video_metadata(video_path)
        total_frames = meta.get("total_frames") or 0
        fps = meta.get("fps") or 30.0
    except Exception as exc:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(exc)})
        await send_ws_message(job_id, {"type":"error", "message": str(exc)})
        return

    # load model runner lazily (import inside to avoid heavy import in startup)
    from app.models.runner import ModelRunner
    model_spec = MODEL_MAP.get(model_name, model_name)
    try:
        runner = ModelRunner(model_spec)
    except Exception as e:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(e)})
        await send_ws_message(job_id, {"type":"error", "message": f"Model load failed: {e}"})
        return

    # open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await redis.hset(meta_key, mapping={"status": "failed", "error": "cannot open video"})
        await send_ws_message(job_id, {"type":"error", "message":"Cannot open video"})
        return

    start_ts = now_ts()
    t_pre, t_inf, t_post = [], [], []
    processed = 0
    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # pre-process timing
            t0 = now_ts()
            # (Here we could resize / convert, but keep original)
            t_pre.append(now_ts() - t0)

            # inference
            t0 = now_ts()
            try:
                res = runner.predict(frame)  # returns dict with 'result_frame'
            except Exception as e:
                # log and continue
                res = {"result_frame": frame}
            t_inf.append(now_ts() - t0)

            # postprocess (encode to jpeg)
            t0 = now_ts()
            out_frame = res.get("result_frame", frame)
            # encode
            success, jpg = cv2.imencode(".jpg", out_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                # skip frame if cannot encode
                frame_idx += 1
                continue
            jpg_bytes = jpg.tobytes()
            # store in redis list
            await redis.rpush(frames_key, jpg_bytes)
            # set TTL
            await redis.expire(frames_key, REDIS_TTL_SECONDS)
            await redis.hset(meta_key, mapping={"processed_frames": frame_idx+1, "total_frames": total_frames})
            await redis.expire(meta_key, REDIS_TTL_SECONDS)

            # send frame to websocket clients as base64 JSON (simple and widely supported)
            b64 = base64.b64encode(jpg_bytes).decode("ascii")
            await send_ws_frame_b64(job_id, b64, frame_idx)

            t_post.append(now_ts() - t0)

            # send progress message
            pct = ((frame_idx+1)/total_frames*100.0) if total_frames else 0.0
            await send_ws_message(job_id, {"type":"progress", "frame": frame_idx+1, "total_frames": total_frames, "pct": round(pct, 2)})

            frame_idx += 1
            processed += 1
            # yield control
            await asyncio.sleep(0)  # cooperative

    except Exception as e:
        await redis.hset(meta_key, mapping={"status": "failed", "error": str(e)})
        await send_ws_message(job_id, {"type":"error", "message": str(e)})
    finally:
        cap.release()

    end_ts = now_ts()
    metrics = compute_metrics(t_pre, t_inf, t_post, start_ts, end_ts, processed, model_name)
    # save final meta
    await redis.hset(meta_key, mapping={
        "status": "done",
        "processed_frames": processed,
        "total_frames": total_frames,
        "start_ts": start_ts,
        "end_ts": end_ts,
        **{k:str(v) for k,v in metrics.items()}
    })
    await redis.expire(meta_key, REDIS_TTL_SECONDS)
    await redis.expire(frames_key, REDIS_TTL_SECONDS)

    # send done message with metrics
    await send_ws_message(job_id, {"type":"done", "metrics": metrics})
