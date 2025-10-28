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
    YOLO model runner supporting:
    - local .pt model loading with safe torch load
    - dynamic color map injection from frontend or Redis
    """

    def __init__(self, model_spec: str, class_colors: dict = None):
        base_dir = os.path.dirname(__file__)
        weights_dir = os.path.join(base_dir, "weights")
        local_path = os.path.join(weights_dir, model_spec)

        # ✅ Try to load local or built-in model
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

        # ✅ Apply color map from argument (if provided)
        self.class_colors = class_colors or {}
        if self.class_colors:
            print(f"🎨 Loaded class colors: {self.class_colors}")
        else:
            print("🎨 No custom class colors provided — using default palette.")

    # ============================================================
    # 🧩 Helper Methods
    # ============================================================
    def _ensure_bgr_tuple(self, color):
        """Convert hex ('#FF0000') or RGB dict to BGR tuple."""
        if isinstance(color, str) and color.startswith("#"):
            color = color.lstrip("#")
            rgb = tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))
            return (rgb[2], rgb[1], rgb[0])  # Convert RGB→BGR
        elif isinstance(color, (list, tuple)) and len(color) == 3:
            return tuple(map(int, color))
        elif isinstance(color, dict):
            return (color.get("b", 0), color.get("g", 0), color.get("r", 0))
        return (0, 255, 0)  # fallback green

    # ============================================================
    # 🔮 Inference
    # ============================================================
    def predict(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        preds = self.model.predict(source=rgb, imgsz=640, conf=0.25, verbose=False)
        out = frame.copy()

        # ✅ Load class name map
        names = getattr(self.model, "names", {})

        # ✅ Default fallback palette
        default_palette = [
            (255, 99, 132),   # pink/red
            (54, 162, 235),   # blue
            (255, 206, 86),   # yellow
            (75, 192, 192),   # teal
            (153, 102, 255),  # purple
            (255, 159, 64),   # orange
            (46, 204, 113),   # green
            (52, 73, 94),     # dark gray
        ]

        if len(preds) > 0:
            res = preds[0]
            boxes = getattr(res, "boxes", None)

            if boxes is not None:
                for b in boxes:
                    try:
                        xyxy = b.xyxy[0].cpu().numpy()
                        conf = float(b.conf[0].cpu().numpy()) if hasattr(b, "conf") else 0.0
                        cls = int(b.cls[0].cpu().numpy()) if hasattr(b, "cls") else -1
                        label_name = names.get(cls, f"class_{cls}")
                        label = f"{label_name} {conf:.2f}"
                        x1, y1, x2, y2 = map(int, xyxy.tolist())

                        # 🎨 Use selected color if available, otherwise fallback
                        if label_name in self.class_colors:
                            color = self._ensure_bgr_tuple(self.class_colors[label_name])
                        else:
                            color = default_palette[cls % len(default_palette)]

                        # 🟩 Draw thicker bounding box
                        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)

                        # 🔠 Larger, more readable label text
                        font_scale = 1.3
                        font_thickness = 3
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)

                        pad_y, pad_x = 10, 12
                        top_left = (x1, max(0, y1 - text_h - pad_y - baseline))
                        bottom_right = (x1 + text_w + pad_x, y1)

                        cv2.rectangle(out, top_left, bottom_right, color, -1)
                        cv2.putText(out, label, (x1 + 5, y1 - 10),
                                    font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

                    except Exception as e:
                        print(f"⚠️ Error drawing box: {e}")
                        continue

        return {"result_frame": out}
