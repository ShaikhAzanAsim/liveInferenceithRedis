import os
import cv2
import torch
import numpy as np
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
                self.model = YOLO(local_path)
            except Exception as e:
                print(f"⚠️ Model load failed with default loader: {e}")
                print("🔁 Retrying with safe load...")
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
            names = getattr(self.model, "names", {})

            if boxes is not None:
                for b in boxes:
                    try:
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if hasattr(b, "cls") else -1
                        label_name = names.get(cls, f"class_{cls}")
                        label = f"{label_name} {conf:.2f}"

                        x1, y1, x2, y2 = map(int, xyxy.tolist())

                        # 🎨 Color per class (consistent)
                        rng = np.random.default_rng(cls)
                        color = tuple(int(x) for x in rng.integers(60, 255, size=3))

                        # 🟩 Draw thicker bounding box
                        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

                        # 🔠 Larger label font
                        font_scale = 1.6
                        font_thickness = 4
                        font = cv2.FONT_HERSHEY_SIMPLEX

                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                        # 🧱 Add padding around label background
                        pad_y = 10
                        pad_x = 10
                        top_left = (x1, y1 - text_h - pad_y - baseline)
                        bottom_right = (x1 + text_w + pad_x, y1)

                        cv2.rectangle(out, top_left, bottom_right, color, -1)
                        cv2.putText(out, label, (x1 + 3, y1 - 6),
                                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                    except Exception:
                        continue

        return {"result_frame": out}
