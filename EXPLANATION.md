# DrowsGuard — Complete Project Explanation

> A reference document for presentations, interviews, and professor demos.

---

## 🎯 What Does This Project Do?

DrowsGuard is a **real-time drowsiness detection system**. It monitors a person's eyes through a webcam and raises an alarm when it detects that the person is falling asleep.

**Most practical use case:** Driver drowsiness detection — one of the leading causes of road accidents worldwide.

**How it feels to use:** Open the browser, click Start, and the system immediately begins watching your eyes. Close your eyes for half a second — alert triggers. Open them — alert clears. All in real time, no special hardware required.

---

## 🏗️ Overall Architecture

```
Webcam (Browser)
      │
      │  JPEG frame encoded as base64
      ▼
WebSocket Connection  ──────────────────────────────────┐
      │                                                  │
      ▼                                                  │
FastAPI Backend (Python)                                 │
      │                                                  │
      ├── dlib HOG detector      → finds the face        │
      ├── dlib shape predictor   → maps 68 landmarks     │
      ├── EAR algorithm          → measures eye openness │
      └── Drowsiness logic       → decides alert or not  │
      │                                                  │
      │  Annotated frame + metrics JSON                  │
      └──────────────────────────────────────────────────┘
      │
      ▼
Browser Frontend (HTML/JS)
      ├── Renders annotated frame inside circular canvas
      ├── Updates EAR meter, stats, event log
      ├── Triggers iris ring animation on alert
      └── Triggers continuous audio beep via Web Audio API
```

---

## 🔬 The Core Algorithm — EAR

The heart of the project is the **Eye Aspect Ratio (EAR)** algorithm from a 2016 research paper by Soukupová and Čech — the same method used in academic drowsiness detection research.

### Step 1 — Detect 68 Facial Landmarks

dlib maps **68 precise landmark points** on the face. Points **36–41** cover the right eye, points **42–47** cover the left eye.

```
        P2   P3
   P1 ──────────── P4     ← horizontal distance (eye width)
        P6   P5
          ↕
     vertical distances
```

### Step 2 — Apply the EAR Formula

```
EAR = (||P2−P6|| + ||P3−P5||) / (2 × ||P1−P4||)
```

This computes the **ratio of vertical eye height to horizontal eye width.**

| Eye State | EAR Value |
|---|---|
| Fully open | ~0.30 |
| Slowly closing | ~0.20–0.25 |
| Closed | < 0.20 |

Both eyes are averaged for stability:
```python
EAR = (left_EAR + right_EAR) / 2
```

### Step 3 — Drowsiness Decision

```python
if EAR < threshold for 10 consecutive frames:
    → Person is DROWSY → trigger ALERT
else:
    → Reset counter → clear alert
```

At ~20 FPS, 10 frames = **0.5 seconds** of closed eyes = drowsiness confirmed.

A normal blink lasts ~150ms (~3 frames) — too short to trigger the alert. This is the key distinction between blinking and drowsiness.

---

## 🔧 Tech Stack — Each Piece Explained

### `detector.py` — The Brain

```python
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
```

| Component | What it does |
|---|---|
| `get_frontal_face_detector()` | HOG-based face detector — scans image with a sliding window, checks if gradient patterns match a face |
| `shape_predictor` | Pre-trained regression tree model — takes face region, outputs 68 landmark coordinates |
| Frame skip (every 5th) | Face detection runs every 5 frames — 5x faster with minimal accuracy loss |
| EAR smoothing | Rolling average of last 5 EAR values — eliminates false alerts from single noisy frames |

**Performance optimization:**
```python
# Expensive face detection — only every 5 frames
if self.frame_count % 5 == 0:
    faces = detector(gray, 0)
    self.last_face = max(faces, ...)   # cache result

# Fast landmark detection — every frame using cached face
shape = predictor(gray, cached_face)
```

**EAR smoothing:**
```python
self.ear_history.append(ear)
if len(self.ear_history) > 5:
    self.ear_history.pop(0)
ear = np.mean(self.ear_history)   # rolling average
```

---

### `main.py` — The Server

```python
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
```

| Component | Why it's used |
|---|---|
| FastAPI | Modern async Python web framework |
| WebSocket | Persistent connection — unlike HTTP which opens/closes per request |
| Response-gated loop | Frontend sends next frame only after receiving previous response — zero queue, zero lag |

**Why WebSocket over HTTP?**
```
HTTP (bad for this):   Open → Request → Response → Close → Open → ...
WebSocket (used here): Open ──────── continuous two-way stream ────────
```

---

### `index.html` — The Interface

| Feature | How it works |
|---|---|
| Circular video | `border-radius: 50%` clips canvas to a circle |
| Iris rings | CSS `animation: rotate` — turn red and speed up on alert |
| Animated background | Canvas API draws slow-moving radial gradient orbs |
| Audio alert | Web Audio API — 740Hz sine wave oscillator, no external file needed |
| Custom cursor | CSS + JS mouse tracking, golden glowing dot |

**Frame loop — the most important part:**
```javascript
ws.onmessage = e => {
  renderFrame(data.frame);      // display annotated frame
  updateMetrics(data.metrics);  // update all UI panels
  if (running) sendFrame();     // send NEXT frame only NOW
};
```
This ensures exactly **one frame is in-flight at any time** — preventing queue buildup which causes lag.

---

## 📊 Complete Data Flow

```
BROWSER                              BACKEND
───────                              ───────
1. Capture webcam frame (640×480)
2. Draw to hidden canvas
3. Encode as JPEG base64 (0.65 quality)
4. Send via WebSocket JSON ──────────►
                                     5.  Decode base64 → numpy array
                                     6.  Convert to grayscale
                                     7.  Detect face (every 5th frame)
                                     8.  Map 68 landmarks
                                     9.  Extract 6+6 eye points
                                     10. Calculate left EAR + right EAR
                                     11. Average → smooth → check threshold
                                     12. Draw eye contour dots
                                     13. Encode frame → base64
                             ◄──────  14. Send frame + metrics JSON
15. Decode frame → draw on canvas
16. Update EAR meter, status, log
17. If alert → rings turn red + beep
18. Send next frame ─────────────────►
```

---

## ⚙️ Settings Reference

| Setting | Default | Meaning |
|---|---|---|
| EAR Threshold | 0.20 | Below this value = eye considered closed |
| Sensitivity | 10 frames | Consecutive closed frames before alert fires |
| JPEG Quality | 0.65 | Optimized for CPU-only — smaller = faster |
| Capture Size | 640×480 | Sufficient for face detection, light on CPU |

---

## 🌟 What Makes This Stand Out

| Feature | Why It Matters |
|---|---|
| Real EAR algorithm | Same method used in published research papers |
| dlib 68 landmarks | Research-grade accuracy — not just bounding boxes |
| FastAPI + WebSocket | Modern, production-grade real-time architecture |
| Response-gated loop | Eliminates lag completely — no frame queue buildup |
| Frame skip (every 5) | 5× faster face detection without accuracy loss |
| EAR smoothing | Eliminates false alerts caused by natural blinking |
| Full stack project | Backend + Frontend + Real browser deployment |
| No GPU required | Runs on any laptop with just a webcam |

---

## 🗣️ Summary

> *"DrowsGuard watches your eyes through a webcam in real time. It uses a library called dlib to locate 68 precise points on your face. From those points, it calculates a value called EAR — Eye Aspect Ratio — which mathematically measures how open your eyes are. A normal blink is too quick to trigger anything. But if your EAR drops below the threshold and stays there for about half a second, the system concludes you're drowsy and immediately fires a visual and audio alert. Everything runs in the browser — no app to install, no GPU needed, just a laptop and a webcam."*

---

## 📁 File Reference

| File | Role |
|---|---|
| `backend/detector.py` | Face detection, 68 landmarks, EAR algorithm, drowsiness logic |
| `backend/main.py` | FastAPI server, WebSocket endpoint, frame encode/decode pipeline |
| `frontend/index.html` | Complete UI — webcam capture, frame loop, alerts, audio |
| `backend/requirements.txt` | All Python dependencies |
| `backend/models/shape_predictor_68_face_landmarks.dat` | Pre-trained dlib model (60MB) — not in repo, downloaded on setup |

---

*Algorithm reference: Soukupová, T. & Čech, J. (2016). Real-Time Eye Blink Detection using Facial Landmarks. 21st Computer Vision Winter Workshop.*
