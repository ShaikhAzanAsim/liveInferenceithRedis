from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"  # for mounting
UPLOAD_DIR = BASE_DIR / "uploads"
TMP_DIR = BASE_DIR / "tmp"

# ensure dirs exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
REDIS_TTL_SECONDS = 1800  # 30 minutes

MAX_UPLOAD_SIZE = 200 * 1024 * 1024  # 200 MB
ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv"}

# ffmpeg binary
FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")

# Model weight paths (example). Update with real paths/weights.
MODEL_MAP = {
    "yolov8n": "yolov8n.pt",   # if ultralytics is installed it resolves, else put path to .pt
    "yolo11n": "yolo11n.pt",   # provide path if not available via package
}
