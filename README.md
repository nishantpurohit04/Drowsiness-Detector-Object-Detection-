# DrowsGuard — AI Drowsiness Detection System

Real-time driver drowsiness detection using **dlib 68-point facial landmarks** and the **Eye Aspect Ratio (EAR)** algorithm. Built with a FastAPI backend, WebSocket streaming, and a live monitoring dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![dlib](https://img.shields.io/badge/dlib-20.0-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Demo

> Webcam → dlib Face Detection → 68 Landmark Mapping → EAR Calculation → Real-time Alert

---

## How It Works

### Eye Aspect Ratio (EAR) Algorithm

The system uses the **EAR algorithm** — the gold standard approach used in drowsiness detection research papers.

```
        P2   P3
   P1 ──────────── P4     ← horizontal distance
        P6   P5

EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

| EAR Value | Eye State |
|---|---|
| ≈ 0.30 | Eyes fully open |
| 0.20 – 0.25 | Eyes closing |
| < 0.20 | Eyes closed → alert triggered |

If EAR stays below **0.20** for **10 consecutive frames** → Drowsiness alert triggered.

---

## Features

- 👁 **68-point facial landmark detection** using dlib — research paper accuracy
- 📊 **Real-time EAR meter** with color-coded status (green / orange / red)
- 🔔 **Continuous audio beep alert** via Web Audio API — no external files needed
- 🎯 **HUD overlay** with EAR bar, L/R individual eye values, countdown bar
- 🚨 **Red vignette border** on drowsiness detection
- 📈 **Session stats** — total alerts, duration, closed frames counter
- 📝 **Alert log** with timestamps
- ⚙️ **Adjustable settings** — EAR threshold and alert sensitivity sliders
- ⚡ **Optimized performance** — face detection runs every 5 frames for smooth video
- 🔄 **WebSocket streaming** — one frame in, one frame out, zero queue buildup

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Face Detection | dlib HOG detector | Fast frontal face detection |
| Landmark Detection | dlib shape predictor | 68 precise facial landmarks |
| EAR Calculation | SciPy euclidean distance | Real EAR formula |
| Frame Processing | OpenCV | Image processing + HUD drawing |
| Backend | FastAPI + WebSockets | Real-time frame streaming |
| Frontend | HTML + CSS + Vanilla JS | Live monitoring dashboard |
| Audio Alert | Web Audio API | Continuous beep — no file needed |

---

## Project Structure

```
drowsguard/
│
├── backend/
│   ├── main.py              # FastAPI app + WebSocket endpoint
│   ├── detector.py          # dlib EAR algorithm + HUD overlay
│   ├── requirements.txt     # Python dependencies
│   └── models/
│       └── shape_predictor_68_face_landmarks.dat  # dlib model (download separately)
│
├── frontend/
│   └── index.html           # Complete dashboard UI
│
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.11
- Webcam
- Chrome browser (recommended)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/drowsguard.git
cd drowsguard
```

### 2. Create virtual environment with Python 3.11
```bash
cd backend
py -3.11 -m venv venv

# Windows
.\venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install fastapi uvicorn websockets opencv-python numpy scipy python-multipart dlib-bin
```

### 4. Download dlib model file
```bash
mkdir models
python -c "import urllib.request; print('Downloading...'); urllib.request.urlretrieve('https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat', 'models/shape_predictor_68_face_landmarks.dat'); print('Done!')"
```

### 5. Run the server
```bash
python main.py
```

### 6. Open the app
Go to **http://localhost:8000** in Chrome → click **▶ Start Monitoring**

---

## Usage

1. Click **▶ Start Monitoring** and allow webcam access
2. Position your face in front of the camera
3. The system will track your eyes in real time
4. If eyes stay closed for ~0.67 seconds → **alert triggers**
5. Adjust **EAR Threshold** and **Alert Sensitivity** sliders as needed
6. Click **↺ Reset Session** to clear stats and start fresh

---

## Configuration

| Setting | Default | Description |
|---|---|---|
| EAR Threshold | 0.25 | Below this = eye considered closed |
| Alert Sensitivity | 20 frames | Consecutive closed frames before alert |

---

## Performance

- **~20 FPS** on CPU (no GPU required)
- Face detection runs every **5 frames** — 5x faster than per-frame detection
- Landmark detection runs **every frame** — maintains accuracy
- WebSocket uses **response-gated** frame sending — zero queue buildup

---

## Requirements

```
fastapi==0.110.0
uvicorn==0.29.0
websockets==12.0
opencv-python==4.9.0.80
numpy==1.26.4
scipy==1.13.0
python-multipart==0.0.9
dlib-bin
```

---

## License

MIT
