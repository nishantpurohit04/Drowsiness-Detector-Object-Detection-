"""
detector.py — Drowsiness detection using dlib 68 facial landmarks
Optimized: face detection runs every 5 frames for smooth performance
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# ── EYE LANDMARK INDICES ──────────────────────────────────────────────────────
LEFT_EYE_IDX  = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

# ── THRESHOLDS ────────────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20


class DrowsinessDetector:
    def __init__(self):
        self.closed_frame_count = 0
        self.alert_active       = False
        self.total_alerts       = 0
        self.ear_history        = []
        self.frame_count        = 0      # tracks frames for skip logic
        self.last_face          = None   # caches last detected face

    def _get_eye_points(self, shape, eye_indices):
        return np.array(
            [(shape.part(i).x, shape.part(i).y) for i in eye_indices],
            dtype=np.float64
        )

    def _calculate_ear(self, eye_points):
        v1  = dist.euclidean(eye_points[1], eye_points[5])
        v2  = dist.euclidean(eye_points[2], eye_points[4])
        h   = dist.euclidean(eye_points[0], eye_points[3])
        return (v1 + v2) / (2.0 * h)

    def _draw_eye_contour(self, frame, eye_points, color):
        pts = eye_points.astype(np.int32)
        for i in range(len(pts)):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), color, 1)
        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, color, -1)

    def process_frame(self, frame, ear_threshold=None, closed_frames=None):
        threshold   = ear_threshold or EAR_THRESHOLD
        frame_limit = closed_frames or CLOSED_FRAMES

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ear           = 0.0
        drowsy        = False
        face_detected = False

        # ── FACE DETECTION — runs every 5 frames only ─────────────────────────
        self.frame_count += 1

        if self.frame_count % 5 == 0 or self.last_face is None:
            # Full detection every 5th frame
            # upsample=0 → faster, good enough for normal webcam distance
            faces = detector(gray, 0)
            if len(faces) > 0:
                self.last_face = max(faces, key=lambda f: f.width() * f.height())
            else:
                self.last_face = None

        # Use cached face for frames in between
        face = self.last_face

        if face is not None:
            face_detected = True

            # Draw face rectangle
            cv2.rectangle(frame,
                          (face.left(),  face.top()),
                          (face.right(), face.bottom()),
                          (0, 200, 255), 2)

            # ── 68 LANDMARKS ─────────────────────────────────────────────────
            shape = predictor(gray, face)

            left_eye_pts  = self._get_eye_points(shape, LEFT_EYE_IDX)
            right_eye_pts = self._get_eye_points(shape, RIGHT_EYE_IDX)

            # ── REAL EAR CALCULATION ──────────────────────────────────────────
            left_ear  = self._calculate_ear(left_eye_pts)
            right_ear = self._calculate_ear(right_eye_pts)
            ear       = (left_ear + right_ear) / 2.0

            # Smooth with rolling average
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)
            ear = float(np.mean(self.ear_history))

            # Draw eye contours
            eye_color = (0, 255, 0) if ear >= threshold else (0, 0, 255)
            self._draw_eye_contour(frame, left_eye_pts,  eye_color)
            self._draw_eye_contour(frame, right_eye_pts, eye_color)

            # ── DROWSINESS LOGIC ──────────────────────────────────────────────
            if ear < threshold:
                self.closed_frame_count += 1
            else:
                self.closed_frame_count = 0
                self.alert_active       = False

            if self.closed_frame_count >= frame_limit:
                drowsy = True
                if not self.alert_active:
                    self.alert_active  = True
                    self.total_alerts += 1

            # ── DRAW INFO ─────────────────────────────────────────────────────
            ear_color = (0, 255, 0) if ear >= threshold else (0, 0, 255)
            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
            cv2.putText(frame, f"L: {left_ear:.2f}  R: {right_ear:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            state_text  = "EYES OPEN"   if ear >= threshold else "EYES CLOSED"
            state_color = (0, 255, 0)   if ear >= threshold else (0, 0, 255)
            cv2.putText(frame, state_text, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, state_color, 2)

            if self.closed_frame_count > 0:
                cv2.putText(frame, f"Closed: {self.closed_frame_count}/{frame_limit}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        else:
            self.closed_frame_count = 0
            self.ear_history        = []
            cv2.putText(frame, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        # ── ALERT OVERLAY ─────────────────────────────────────────────────────
        if drowsy and self.alert_active:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.putText(frame, "DROWSINESS ALERT!", (w//2 - 160, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        metrics = {
            "ear"           : round(float(ear), 3),
            "drowsy"        : bool(drowsy),
            "alert_active"  : bool(self.alert_active),
            "total_alerts"  : self.total_alerts,
            "face_detected" : face_detected,
            "closed_frames" : self.closed_frame_count,
            "threshold"     : threshold
        }

        return frame, metrics

    def reset(self):
        self.closed_frame_count = 0
        self.alert_active       = False
        self.total_alerts       = 0
        self.ear_history        = []
        self.frame_count        = 0
        self.last_face          = None