from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pathlib import Path
import io, os, tempfile, shutil, asyncio, subprocess

from app.utils.redis_client import get_redis
from app.config import TMP_DIR, REDIS_TTL_SECONDS, FFMPEG_BIN

router = APIRouter()

@router.get("/download/{job_id}")
async def download_job(job_id: str):
    redis = get_redis()
    frames_key = f"job:{job_id}:frames"
    meta_key = f"job:{job_id}:meta"

    # check meta exists
    meta = await redis.hgetall(meta_key)
    if not meta:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    # get all frames
    frames = await redis.lrange(frames_key, 0, -1)
    if not frames:
        raise HTTPException(status_code=404, detail="No frames cached for job (expired or failed)")

    # write frames to temp folder
    out_dir = TMP_DIR / f"dl_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_files = []
    for i, b in enumerate(frames):
        p = out_dir / f"frame_{i:06d}.jpg"
        with open(p, "wb") as f:
            f.write(b)
        frame_files.append(p)

    # determine fps from meta if present, else default 25
    fps = float(meta.get(b"fps", b"25").decode() ) if b"fps" in meta else 25.0

    out_video = out_dir / f"{job_id}.mp4"

    # call ffmpeg to stitch frames
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-framerate", str(fps),
        "-i", str(out_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_video)
    ]
    # run blocking in thread
    def run_ffmpeg():
        subprocess.run(cmd, check=True)

    try:
        await asyncio.to_thread(run_ffmpeg)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {e}")

    # stream file
    # Optionally clear the redis cache for the job
    await redis.delete(frames_key)
    await redis.delete(meta_key)

    return FileResponse(str(out_video), filename=f"{job_id}.mp4", media_type="video/mp4")
