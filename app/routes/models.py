# app/routes/models.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from ultralytics import YOLO
import torch
import tempfile
import os

router = APIRouter()

@router.post("/analyze_model")
async def analyze_model(model_file: UploadFile = File(...)):
    """Extract class names from uploaded YOLO model (.pt)."""
    if not model_file.filename.endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp:
        contents = await model_file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        model = YOLO(tmp_path)
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            class_names = list(names.values())
        else:
            class_names = names
        os.remove(tmp_path)
        return {"class_names": class_names}
    except Exception as e:
        os.remove(tmp_path)
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")
