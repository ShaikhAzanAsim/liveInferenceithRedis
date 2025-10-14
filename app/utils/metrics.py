import time
from typing import List, Dict

def now_ts() -> float:
    return time.time()

def compute_metrics(t_preprocess: List[float], t_infer: List[float], t_post: List[float], start_ts: float, end_ts: float, total_frames: int, model_name: str) -> Dict:
    total_time = end_ts - start_ts
    avg_fps = total_frames / total_time if total_time > 0 else 0.0
    avg_pre = sum(t_preprocess) / len(t_preprocess) if t_preprocess else 0.0
    avg_inf = sum(t_infer) / len(t_infer) if t_infer else 0.0
    avg_post = sum(t_post) / len(t_post) if t_post else 0.0

    return {
        "model": model_name,
        "total_frames": total_frames,
        "total_time_s": round(total_time, 4),
        "avg_fps": round(avg_fps, 4),
        "avg_preprocess_ms": round(avg_pre * 1000, 3),
        "avg_infer_ms": round(avg_inf * 1000, 3),
        "avg_postprocess_ms": round(avg_post * 1000, 3),
    }
