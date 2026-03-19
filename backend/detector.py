"""
detector.py — Drowsiness detection using dlib 68 facial landmarks
Optimized: face detection runs every 5 frames for smooth performance
Beautiful HUD overlay with modern design
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

# ── COLORS (BGR) ──────────────────────────────────────────────────────────────
COLOR_SAFE    = (0, 230, 118)      # green
COLOR_WARN    = (0, 165, 255)      # orange
COLOR_DANGER  = (60,  20, 220)     # red
COLOR_ACCENT  = (255, 220,  0)     # cyan-ish
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0)
COLOR_PANEL   = (18,  22,  28)     # dark panel bg


class DrowsinessDetector:
    def __init__(self):
        self.closed_frame_count = 0
        self.alert_active       = False
        self.total_alerts       = 0
        self.ear_history        = []
        self.frame_count        = 0
        self.last_face          = None

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

    # ── HUD DRAWING HELPERS ───────────────────────────────────────────────────
    def _draw_rounded_rect(self, frame, x, y, w, h, color, alpha=0.6, radius=8):
        """Draw a semi-transparent rounded rectangle."""
        overlay = frame.copy()
        cv2.rectangle(overlay, (x + radius, y), (x + w - radius, y + h), color, -1)
        cv2.rectangle(overlay, (x, y + radius), (x + w, y + h - radius), color, -1)
        cv2.circle(overlay, (x + radius,     y + radius),     radius, color, -1)
        cv2.circle(overlay, (x + w - radius, y + radius),     radius, color, -1)
        cv2.circle(overlay, (x + radius,     y + h - radius), radius, color, -1)
        cv2.circle(overlay, (x + w - radius, y + h - radius), radius, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def _draw_text(self, frame, text, x, y, scale=0.55, color=COLOR_WHITE, thickness=1):
        cv2.putText(frame, text, (x, y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

    def _draw_ear_bar(self, frame, x, y, w, ear, threshold):
        """Draw a horizontal EAR progress bar."""
        bar_h  = 6
        filled = int(min(ear / 0.40, 1.0) * w)
        thresh = int((threshold / 0.40) * w)

        # Track
        cv2.rectangle(frame, (x, y), (x + w, y + bar_h), (50, 50, 50), -1)

        # Fill color based on EAR
        if ear >= threshold:
            bar_color = COLOR_SAFE
        elif ear >= threshold - 0.05:
            bar_color = COLOR_WARN
        else:
            bar_color = COLOR_DANGER

        # Filled portion
        cv2.rectangle(frame, (x, y), (x + filled, y + bar_h), bar_color, -1)

        # Threshold marker line
        cv2.rectangle(frame, (x + thresh - 1, y - 2),
                      (x + thresh + 1, y + bar_h + 2), COLOR_ACCENT, -1)

    def _draw_hud(self, frame, ear, left_ear, right_ear,
                  threshold, closed_count, frame_limit, face_detected):
        """Draw the main HUD panel in bottom-left corner."""
        h, w = frame.shape[:2]

        panel_x = 12
        panel_y = h - 130
        panel_w = 240
        panel_h = 118

        # Panel background
        self._draw_rounded_rect(frame, panel_x, panel_y,
                                panel_w, panel_h, COLOR_PANEL, alpha=0.72)

        # Panel top accent line
        color = COLOR_SAFE if ear >= threshold else COLOR_DANGER
        cv2.rectangle(frame,
                      (panel_x, panel_y),
                      (panel_x + panel_w, panel_y + 2),
                      color, -1)

        # ── EAR VALUE ────────────────────────────────────────────────────────
        ear_str = f"{ear:.3f}"
        self._draw_text(frame, "EAR", panel_x + 12, panel_y + 22,
                        scale=0.42, color=(160, 160, 160))
        self._draw_text(frame, ear_str, panel_x + 44, panel_y + 22,
                        scale=0.58, color=color, thickness=1)

        # ── LEFT / RIGHT ──────────────────────────────────────────────────────
        self._draw_text(frame,
                        f"L {left_ear:.2f}   R {right_ear:.2f}",
                        panel_x + 12, panel_y + 42,
                        scale=0.38, color=(180, 180, 180))

        # ── EAR BAR ───────────────────────────────────────────────────────────
        self._draw_ear_bar(frame, panel_x + 12, panel_y + 54,
                           panel_w - 24, ear, threshold)

        # ── STATUS ────────────────────────────────────────────────────────────
        if not face_detected:
            status      = "NO FACE"
            status_col  = (120, 120, 120)
        elif ear >= threshold:
            status      = "EYES OPEN"
            status_col  = COLOR_SAFE
        else:
            status      = "EYES CLOSED"
            status_col  = COLOR_DANGER

        self._draw_text(frame, status,
                        panel_x + 12, panel_y + 80,
                        scale=0.52, color=status_col, thickness=1)

        # ── FRAME COUNTER BAR ─────────────────────────────────────────────────
        if closed_count > 0:
            progress = int((closed_count / frame_limit) * (panel_w - 24))
            cv2.rectangle(frame,
                          (panel_x + 12, panel_y + 90),
                          (panel_x + 12 + panel_w - 24, panel_y + 96),
                          (50, 50, 50), -1)
            cv2.rectangle(frame,
                          (panel_x + 12, panel_y + 90),
                          (panel_x + 12 + progress, panel_y + 96),
                          COLOR_WARN, -1)
            self._draw_text(frame,
                            f"DROWSY IN {frame_limit - closed_count} FRAMES",
                            panel_x + 12, panel_y + 112,
                            scale=0.35, color=COLOR_WARN)

    def _draw_alert_overlay(self, frame):
        """Draw red vignette border only — frontend handles the alert text."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        for i, thickness in enumerate([60, 40, 20]):
            alpha = 0.12 + i * 0.04
            cv2.rectangle(overlay, (0, 0), (w, h), COLOR_DANGER, thickness)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def process_frame(self, frame, ear_threshold=None, closed_frames=None):
        threshold   = ear_threshold or EAR_THRESHOLD
        frame_limit = closed_frames or CLOSED_FRAMES

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ear           = 0.0
        left_ear      = 0.0
        right_ear     = 0.0
        drowsy        = False
        face_detected = False

        # ── FACE DETECTION every 5 frames ─────────────────────────────────────
        self.frame_count += 1
        if self.frame_count % 5 == 0 or self.last_face is None:
            faces = detector(gray, 0)
            self.last_face = max(faces, key=lambda f: f.width() * f.height()) \
                             if len(faces) > 0 else None

        face = self.last_face

        if face is not None:
            face_detected = True

            # Face box — thin, accent colored
            cv2.rectangle(frame,
                          (face.left(),  face.top()),
                          (face.right(), face.bottom()),
                          COLOR_ACCENT, 1)

            # Corner brackets instead of full rectangle — cleaner look
            bx, by = face.left(), face.top()
            bw, bh = face.right() - bx, face.bottom() - by
            ln = 16  # bracket length
            for (sx, sy, dx, dy) in [
                (bx, by, 1, 1), (bx+bw, by, -1, 1),
                (bx, by+bh, 1, -1), (bx+bw, by+bh, -1, -1)
            ]:
                cv2.line(frame, (sx, sy), (sx + dx*ln, sy), COLOR_ACCENT, 2)
                cv2.line(frame, (sx, sy), (sx, sy + dy*ln), COLOR_ACCENT, 2)

            # ── LANDMARKS ─────────────────────────────────────────────────────
            shape = predictor(gray, face)
            left_eye_pts  = self._get_eye_points(shape, LEFT_EYE_IDX)
            right_eye_pts = self._get_eye_points(shape, RIGHT_EYE_IDX)

            left_ear  = self._calculate_ear(left_eye_pts)
            right_ear = self._calculate_ear(right_eye_pts)
            ear       = (left_ear + right_ear) / 2.0

            # Smooth
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)
            ear = float(np.mean(self.ear_history))

            eye_color = COLOR_SAFE if ear >= threshold else COLOR_DANGER
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

        else:
            self.closed_frame_count = 0
            self.ear_history        = []

        # ── DRAW HUD ──────────────────────────────────────────────────────────
        self._draw_hud(frame, ear, left_ear, right_ear,
                       threshold, self.closed_frame_count,
                       frame_limit, face_detected)

        # ── DRAW ALERT ────────────────────────────────────────────────────────
        if drowsy and self.alert_active:
            self._draw_alert_overlay(frame)

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