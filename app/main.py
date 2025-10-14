from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.routes import upload, download, ws
from app.config import STATIC_DIR

app = FastAPI(title="FastAPI Video Inference")

# CORS (if serving frontend separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routes
app.include_router(upload.router, prefix="")
app.include_router(download.router, prefix="")
app.include_router(ws.router, prefix="")

# serve static frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
