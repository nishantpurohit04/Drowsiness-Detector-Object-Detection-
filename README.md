# DrowsGuard — AI Drowsiness Detection System

A real-time drowsiness detection system using computer vision and facial landmark analysis. Built with MediaPipe, FastAPI, and a WebSocket-powered frontend.

---

## Demo

> Webcam feed → Face Mesh → EAR calculation → Real-time alert

---

## How It Works

The system uses the **Eye Aspect Ratio (EAR)** algorithm:

```
EAR = (||P2-P6|| + ||P3-P5||) / (2 × ||P1-P4||)
```

- **EAR ≈ 0.30** → Eyes open
- **EAR < 0.18** → Eyes closing
- **EAR < 0.18 for 20+ consecutive frames** → Drowsiness alert triggered

---

## Tech Stack

| Layer | Technology |
|---|---|
| Face Landmark Detection | MediaPipe Face Mesh (468 landmarks) |
| EAR Calculation | Custom Python + SciPy |
| Backend | FastAPI + WebSockets |
| Frontend | HTML + CSS + Vanilla JS |
| Real-time Communication | WebSocket (binary frame streaming) |

---

## Project Structure

```
drowsguard/
├── backend/
│   ├── main.py              # FastAPI app + WebSocket endpoint
│   ├── detector.py          # EAR algorithm + drowsiness logic
│   └── requirements.txt
├── frontend/
│   └── index.html           # Complete UI
└── README.md
```

---

## Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/drowsguard.git
cd drowsguard
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Run the server
```bash
python main.py
```

### 5. Open the app
Go to **http://localhost:8000** in your browser and click **Start Monitoring**.

---

## Features

- Real-time EAR score meter with color coding
- Visual alert overlay when drowsiness detected
- Audio beep alert (Web Audio API)
- Session stats — total alerts, duration, status
- Alert log with timestamps
- Adjustable EAR threshold and sensitivity settings
- FPS counter

---

## Configuration

You can adjust these in the UI:

| Setting | Default | Description |
|---|---|---|
| EAR Threshold | 0.18 | Below this value = eye considered closed |
| Alert Sensitivity | 20 frames | Consecutive closed frames before alert |

---

## Requirements

- Python 3.8+
- Webcam
- Modern browser (Chrome recommended)

---

## License

MIT
