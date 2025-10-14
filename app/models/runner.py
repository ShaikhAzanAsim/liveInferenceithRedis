import os
import cv2
import torch
from ultralytics import YOLO

# ✅ Safe-load fix for PyTorch 2.6+
try:
    from ultralytics.nn.tasks import DetectionModel
    torch.serialization.add_safe_globals([DetectionModel])
except Exception as e:
    print(f"⚠️ Safe global registration failed: {e}")

class ModelRunner:
    """
    YOLO model runner supporting local .pt models with safe torch load.
    """

    def __init__(self, model_spec: str):
        base_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(base_dir, "weights")
        local_path = os.path.join(weights_dir, model_spec)

        if os.path.exists(local_path):
            print(f"🔹 Loading local model: {local_path}")
            try:
                # Try normal YOLO load
                self.model = YOLO(local_path)
            except Exception as e:
                print(f"⚠️ Model load failed with default loader: {e}")
                print("🔁 Retrying with safe load...")
                # Fallback manual torch load
                ckpt = torch.load(local_path, weights_only=False, map_location="cpu")
                self.model = YOLO(model=ckpt)
        else:
            print(f"🔹 Loading built-in YOLO model: {model_spec}")
            self.model = YOLO(model_spec)

    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.model.predict(source=rgb, imgsz=640, conf=0.25, verbose=False)
        out = frame.copy()

        if len(preds) > 0:
            res = preds[0]
            boxes = getattr(res, "boxes", None)
            if boxes is not None:
                for b in boxes:
                    try:
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else None
                        cls = int(b.cls[0].cpu().numpy()) if hasattr(b, "cls") else None
                        label = f"{cls}:{conf:.2f}" if cls is not None else f"{conf:.2f}"
                        x1, y1, x2, y2 = map(int, xyxy.tolist())
                        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(out, label, (x1, max(y1 - 6, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    except Exception:
                        continue

        return {"result_frame": out}
