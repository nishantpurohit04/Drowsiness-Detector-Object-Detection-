"""
main.py — FastAPI backend for DrowsGuard
Receives video frames from the browser via WebSocket,
processes them through the drowsiness detector,
and streams back annotated frames + metrics.
"""

import cv2
import base64
import json
import numpy as np
from fastapi             import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses   import FileResponse
import os

from detector import DrowsinessDetector


# ── APP SETUP ────────────────────────────────────────────────────────────────
app = FastAPI(title="DrowsGuard API", version="1.0.0")

# Allow frontend (running on a different port) to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# Serve frontend files
frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend')
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


# ── DETECTOR INSTANCE ────────────────────────────────────────────────────────
# One shared detector per server instance
# In production you'd create one per connected client
detector = DrowsinessDetector()


# ── ROUTES ───────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    """Serve the frontend HTML page."""
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
async def health():
    """Health check — useful for verifying server is running."""
    return {"status": "ok", "message": "DrowsGuard is running"}


@app.post("/reset")
async def reset_session():
    """Reset the detector state — called when user starts a new session."""
    detector.reset()
    return {"status": "reset", "message": "Session reset successfully"}


# ── WEBSOCKET ENDPOINT ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Main WebSocket connection.

    Flow:
    1. Frontend captures webcam frame
    2. Converts to base64 and sends as JSON
    3. Backend decodes → processes → encodes result
    4. Sends back annotated frame + metrics as JSON
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            # ── RECEIVE FRAME FROM FRONTEND ──────────────────────────────────
            data = await websocket.receive_text()
            payload = json.loads(data)

            # Extract base64 image and optional settings
            img_b64       = payload.get("frame", "")
            ear_threshold = payload.get("ear_threshold", 0.18)
            closed_frames = payload.get("closed_frames", 20)

            if not img_b64:
                continue

            # ── DECODE BASE64 → OPENCV FRAME ─────────────────────────────────
            # base64 string → bytes → numpy array → BGR image
            img_bytes = base64.b64decode(img_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # ── PROCESS THROUGH DETECTOR ──────────────────────────────────────
            annotated_frame, metrics = detector.process_frame(
                frame,
                ear_threshold = ear_threshold,
                closed_frames = closed_frames
            )

            # ── ENCODE ANNOTATED FRAME BACK TO BASE64 ────────────────────────
            # Encode as JPEG (smaller than PNG → faster WebSocket transfer)
            _, buffer   = cv2.imencode('.jpg', annotated_frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_b64   = base64.b64encode(buffer).decode('utf-8')

            # ── SEND RESPONSE ─────────────────────────────────────────────────
            response = {
                "frame"   : frame_b64,
                "metrics" : metrics
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


# ── RUN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print("Starting DrowsGuard server on http://localhost:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
