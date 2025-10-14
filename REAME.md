# 🎥 YOLO Live Video Inference Web App (FastAPI + Redis)

## 🚀 Overview

This project is a **real-time video inference platform** built with **FastAPI**, **Redis**, and **YOLOv8/YOLO11** models.  
Users can upload a video, select their preferred YOLO model, and watch **frame-by-frame live inference** directly on the web interface.  
All inference frames are temporarily stored in **Redis cache** for up to **30 minutes** or until the user downloads the final video.

---

## ✨ Key Features

- 🧠 **Model Selection:** Choose between `yolov8n` and `yolo11n` models (local `.pt` weights supported).  
- ⚡ **Live Inference:** Displays frame-by-frame inference progress in real time.  
- 🕓 **Progress Bar:** Visual progress tracking for ongoing inference jobs.  
- 💾 **Redis Cache Storage:** Stores inference frames and metadata temporarily for 30 minutes.  
- 📊 **Performance Metrics:** Shows average FPS, total inference time, and model details after completion.  
- 📥 **Downloadable Video:** Once inference completes, users can download the fully processed video.  
- 🔒 **Isolated Jobs:** Each video inference runs independently using a unique job ID.

---

## 🧩 Tech Stack

| Component | Description |
|------------|--------------|
| **Backend** | [FastAPI](https://fastapi.tiangolo.com/) |
| **Frontend** | HTML, CSS, Vanilla JavaScript |
| **Model Inference** | [Ultralytics YOLO](https://docs.ultralytics.com/) (v8 & v11 supported) |
| **Cache / Job Queue** | [Redis](https://redis.io/) |
| **Video Encoding** | [FFmpeg](https://ffmpeg.org/) |

---

## 📂 Project Structure

liveInferenceWithRedis/
├── app/
│ ├── main.py # FastAPI app entry point
│ ├── routes/
│ │ ├── inference_router.py
│ │ └── upload_router.py
│ ├── utils/
│ │ ├── runner.py # YOLO inference handler
│ │ ├── redis_client.py # Redis cache interface
│ │ └── ffmpeg_utils.py # Video encoding utilities
│ ├── models/
│ │ └── weights/
│ │ ├── yolov8n.pt
│ │ └── yolo11n.pt
│ ├── static/
│ │ ├── styles.css
│ │ └── app.js
│ └── templates/
│ └── index.html
├── tmp/ # Temporary job directories for frames/videos
├── .gitignore
├── requirements.txt
└── README.md


---

## ⚙️ Setup Instructions

### 1️ Clone the Repository

git clone https://github.com/yourusername/liveInferenceWithRedis.git
cd liveInferenceWithRedis

### 2 Virtual Env
python -m venv venv
venv\Scripts\activate   # On Windows
# or
source venv/bin/activate  # On Linux/Mac

### 3 Install req
pip install -r requirements.txt

### 4 Run Redis Server
redis-server

### 5 Run main.py
uvicorn app.main:app --reload



### You can inspect using Redis CLI:

redis-cli
KEYS *


👨‍💻 Author

Shaikh Azan
💼 AI & Software Engineer
📧 [azanasim1@example.com]
🌐 [https://github.com/ShaikhAzanAsim]