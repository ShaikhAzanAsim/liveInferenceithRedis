import cv2
import os
import uuid
from pathlib import Path
from typing import Tuple

from app.config import UPLOAD_DIR, TMP_DIR, FFMPEG_BIN

def save_upload_to_disk(file_bytes: bytes, filename_hint: str) -> Path:
    suffix = Path(filename_hint).suffix or ".mp4"
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / name
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path

def read_video_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError("Cannot open video for metadata")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"total_frames": total, "fps": fps, "width": width, "height": height}

def stitch_frames_to_video(frame_paths: list, out_path: Path, fps: float) -> None:
    """
    Uses ffmpeg to stitch frames named with increasing indices.
    frame_paths: list of file paths in correct order.
    """
    if not frame_paths:
        raise RuntimeError("No frames to stitch")

    tmp_dir = TMP_DIR / f"ff_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    # copy frames into tmp_dir with 6-digit ordering
    for i, src in enumerate(frame_paths):
        dst = tmp_dir / f"frame_{i:06d}.jpg"
        with open(src, "rb") as fr, open(dst, "wb") as fw:
            fw.write(fr.read())

    # build ffmpeg args
    # -y overwrite, -r fps, -i frame_%06d.jpg
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-framerate", str(fps),
        "-i", str(tmp_dir / "frame_%06d.jpg"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(out_path)
    ]
    import subprocess
    subprocess.run(cmd, check=True)
    # cleanup
    for p in tmp_dir.iterdir():
        p.unlink()
    tmp_dir.rmdir()
