"""
detector.py — Clean version
Only draws eye landmark dots on the frame.
All UI, text, alerts handled by the frontend.
"""

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

LEFT_EYE_IDX  = list(range(42, 48))
RIGHT_EYE_IDX = list(range(36, 42))

EAR_THRESHOLD = 0.25
CLOSED_FRAMES = 20


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
        v1 = dist.euclidean(eye_points[1], eye_points[5])
        v2 = dist.euclidean(eye_points[2], eye_points[4])
        h  = dist.euclidean(eye_points[0], eye_points[3])
        return (v1 + v2) / (2.0 * h)

    def _draw_eye_contour(self, frame, eye_points, color):
        """Draw only subtle eye landmark dots — no boxes, no text."""
        pts = eye_points.astype(np.int32)
        # Connect the 6 points with thin lines
        for i in range(len(pts)):
            cv2.line(frame,
                     tuple(pts[i]),
                     tuple(pts[(i+1) % len(pts)]),
                     color, 1, cv2.LINE_AA)
        # Small dots at each landmark
        for pt in pts:
            cv2.circle(frame, tuple(pt), 2, color, -1, cv2.LINE_AA)

    def process_frame(self, frame, ear_threshold=None, closed_frames=None):
        threshold   = ear_threshold or EAR_THRESHOLD
        frame_limit = closed_frames or CLOSED_FRAMES

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ear           = 0.0
        left_ear      = 0.0
        right_ear     = 0.0
        drowsy        = False
        face_detected = False

        # Face detection every 5 frames
        self.frame_count += 1
        if self.frame_count % 5 == 0 or self.last_face is None:
            faces = detector(gray, 0)
            self.last_face = max(faces, key=lambda f: f.width() * f.height()) \
                             if len(faces) > 0 else None

        face = self.last_face

        if face is not None:
            face_detected = True
            shape = predictor(gray, face)

            left_eye_pts  = self._get_eye_points(shape, LEFT_EYE_IDX)
            right_eye_pts = self._get_eye_points(shape, RIGHT_EYE_IDX)

            left_ear  = self._calculate_ear(left_eye_pts)
            right_ear = self._calculate_ear(right_eye_pts)
            ear       = (left_ear + right_ear) / 2.0

            # Smooth EAR
            self.ear_history.append(ear)
            if len(self.ear_history) > 5:
                self.ear_history.pop(0)
            ear = float(np.mean(self.ear_history))

            # Eye color: teal when open, red when closed
            if ear >= threshold:
                eye_color = (180, 220, 180)   # soft green-white
            else:
                eye_color = (100, 100, 220)   # soft red

            # Draw ONLY the eye contours — nothing else
            self._draw_eye_contour(frame, left_eye_pts,  eye_color)
            self._draw_eye_contour(frame, right_eye_pts, eye_color)

            # Drowsiness logic
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

        metrics = {
            "ear"           : round(float(ear), 3),
            "left_ear"      : round(float(left_ear), 3),
            "right_ear"     : round(float(right_ear), 3),
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