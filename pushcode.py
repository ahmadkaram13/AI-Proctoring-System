"""
═══════════════════════════════════════════════════════════════════════════════
                        🎓 AI PROCTORING SYSTEM 🎓
═══════════════════════════════════════════════════════════════════════════════

A complete webcam-based proctoring system with:
  • Face tracking (no-face / multiple faces / improved head pose via MediaPipe)
  • Drowsiness detection (MediaPipe Eye Aspect Ratio)
  • Phone detection (YOLOv8)
  • Face recognition (identity verification)
  • Audio / speaking detection (sounddevice RMS)
  • Tab / window-switch detection (pywin32)
  • Session recording (OpenCV VideoWriter)
  • Weighted penalty scoring
  • Thread-safe modern GUI with embedded live feed
  • Event cooldown, screenshots, Excel reports


═══════════════════════════════════════════════════════════════════════════════
"""



import os
import time
import platform
import threading
from datetime import datetime
from tkinter import messagebox

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
import customtkinter as ctk


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░░  OPTIONAL DEPENDENCIES  ░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

try:
    import mediapipe as mp
    _HAS_MEDIAPIPE = True
except ImportError:
    _HAS_MEDIAPIPE = False
    print("⚠  mediapipe not installed — drowsiness & accurate pose disabled")

try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except ImportError:
    _HAS_YOLO = False
    print("⚠  ultralytics not installed — phone detection disabled")

try:
    import face_recognition
    _HAS_FACEREC = True
except ImportError:
    _HAS_FACEREC = False
    print("⚠  face_recognition not installed — identity verification disabled")

try:
    import sounddevice as sd
    _HAS_AUDIO = True
except ImportError:
    _HAS_AUDIO = False
    print("⚠  sounddevice not installed — audio detection disabled")

try:
    import win32gui
    import win32process
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False
    print("⚠  pywin32 not installed — tab-switch detection disabled")


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  CONFIGURATION  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

EAR_THRESHOLD        = 0.25
EAR_CONSEC_FRAMES    = 20
NO_FACE_TIMEOUT      = 3.0
HEAD_LEFT_THRESHOLD  = 0.40
HEAD_RIGHT_THRESHOLD = 0.60
YOLO_CONFIDENCE      = 0.5
FACE_RECOGNITION_TOLERANCE = 0.6
AUDIO_THRESHOLD      = 0.025
AUDIO_COOLDOWN       = 4.0
TAB_POLL_INTERVAL    = 0.5

EVENT_COOLDOWN = 3.0

SCREENSHOT_DIR = "screenshots"
REPORTS_DIR    = "reports"
RECORDINGS_DIR = "recordings"
YOLO_MODEL     = "yolov8n.pt"
CAMERA_INDEX   = 0

FEED_UPDATE_MS = 30
WINDOW_WIDTH   = 920
WINDOW_HEIGHT  = 620

PENALTY_MAP = {
    "No Face Detected":    5,
    "Head Turned Left":    1,
    "Head Turned Right":   1,
    "Multiple Faces":      20,
    "Drowsiness Detected": 5,
    "Phone Detected":      10,
    "Unknown Person":      15,
    "Speaking Detected":   3,
    "Tab Switch Detected": 8,
}
DEFAULT_PENALTY = 2


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  EVENT LOGGER  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

class EventLogger:
    """Thread-safe event logger with per-event cooldown and weighted scoring."""

    def __init__(self):
        self.events = []
        self.warning_count = 0
        self.start_time = time.time()
        self._last_event_time = {}
        self._lock = threading.Lock()
        for d in (SCREENSHOT_DIR, REPORTS_DIR, RECORDINGS_DIR):
            os.makedirs(d, exist_ok=True)

    def log(self, event: str, frame=None, cooldown: float = EVENT_COOLDOWN) -> bool:
        now = time.time()
        with self._lock:
            if now - self._last_event_time.get(event, 0) < cooldown:
                return False
            self._last_event_time[event] = now
            self.warning_count += 1
            penalty = PENALTY_MAP.get(event, DEFAULT_PENALTY)
            self.events.append({
                "Time":        datetime.now().strftime("%H:%M:%S"),
                "Elapsed (s)": round(now - self.start_time, 1),
                "Event":       event,
                "Warning #":   self.warning_count,
                "Penalty":     penalty,
            })

        if frame is not None:
            safe_time = datetime.now().strftime("%H-%M-%S")
            path = os.path.join(
                SCREENSHOT_DIR,
                f"{event.replace(' ', '_')}_{safe_time}.png",
            )
            try:
                cv2.imwrite(path, frame)
            except Exception:
                pass

        self._beep_async()
        return True

    @property
    def score(self) -> int:
        with self._lock:
            total = sum(e["Penalty"] for e in self.events)
        return max(0, 100 - total)

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time

    def _beep_async(self):
        def _play():
            try:
                if platform.system() == "Windows":
                    import winsound
                    winsound.Beep(1000, 200)
                else:
                    print("\a", end="", flush=True)
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

    def export(self):
        if not self.events:
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(REPORTS_DIR, f"report_{timestamp}.xlsx")

        events_df = pd.DataFrame(self.events)
        freq_df = (
            events_df
            .groupby("Event")
            .agg(Count=("Event", "count"), Total_Penalty=("Penalty", "sum"))
            .reset_index()
            .sort_values("Count", ascending=False)
        )
        summary_df = pd.DataFrame([{
            "Session Duration (s)": round(self.elapsed),
            "Total Warnings":       self.warning_count,
            "Total Penalty":        sum(e["Penalty"] for e in self.events),
            "Final Score (%)":      self.score,
            "Unique Event Types":   events_df["Event"].nunique(),
        }])

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary",         index=False)
            freq_df.to_excel(   writer, sheet_name="Event Frequency", index=False)
            events_df.to_excel( writer, sheet_name="Events",          index=False)

        return path

    def reset(self):
        with self._lock:
            self.events.clear()
            self.warning_count = 0
            self.start_time = time.time()
            self._last_event_time.clear()


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░░░  AUDIO MONITOR  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

class AudioMonitor:
    """Detects speaking via microphone RMS and fires a callback."""

    def __init__(self, on_speaking):
        self._on_speaking = on_speaking
        self._stream  = None
        self._running = False

    def start(self):
        if not _HAS_AUDIO:
            return
        self._running = True
        try:
            self._stream = sd.InputStream(
                channels=1, samplerate=16000, blocksize=1024,
                callback=self._cb,
            )
            self._stream.start()
        except Exception as e:
            print(f"⚠  Audio stream error: {e}")

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _cb(self, indata, frames, time_info, status):
        if not self._running:
            return
        rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2)))
        if rms > AUDIO_THRESHOLD:
            self._on_speaking(rms)


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░  TAB SWITCH MONITOR  ░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

class TabSwitchMonitor:
    """Polls foreground window PID; fires callback when focus leaves our app."""

    def __init__(self, on_switch, our_pid: int):
        self._on_switch   = on_switch
        self._our_pid     = our_pid
        self._running     = False
        self._was_focused = True
        self._thread      = None

    def start(self):
        if not _HAS_WIN32:
            return
        try:
            hwnd = win32gui.GetForegroundWindow()
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            self._was_focused = (pid == self._our_pid)
        except Exception:
            self._was_focused = True

        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _poll(self):
        while self._running:
            try:
                hwnd = win32gui.GetForegroundWindow()
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                focused = (pid == self._our_pid)
                if self._was_focused and not focused:
                    self._on_switch()
                self._was_focused = focused
            except Exception:
                pass
            time.sleep(TAB_POLL_INTERVAL)


# ═══════════════════════════════════════════════════════════════════════════
# ░░░░░░░░░░░░░░░░░░░░░░░░░░  DETECTION ENGINE  ░░░░░░░░░░░░░░░░░░░░░░░░░░░
# ═══════════════════════════════════════════════════════════════════════════

class DetectionEngine:
    """Runs all CV detections on a frame and returns structured results."""

    LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
    RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]

    NOSE_TIP    = 1
    LEFT_CHEEK  = 234
    RIGHT_CHEEK = 454

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        self.face_mesh = None
        if _HAS_MEDIAPIPE:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=3, refine_landmarks=True,
                min_detection_confidence=0.5, min_tracking_confidence=0.5,
            )

        self.yolo = None
        if _HAS_YOLO:
            try:
                self.yolo = YOLO(YOLO_MODEL)
            except Exception as e:
                print(f"⚠  YOLO failed to load: {e}")

        self.ear_consec_count  = 0
        self.enrolled_encoding = None

    def enroll_face(self, frame) -> bool:
        if not _HAS_FACEREC:
            return False
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        encs = face_recognition.face_encodings(rgb)
        if encs:
            self.enrolled_encoding = encs[0]
            return True
        return False

    @staticmethod
    def _ear(eye_indices, landmarks, w, h) -> float:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C > 0 else 0.0

    def analyze_frame(self, frame):
        """Run all detections. Returns (results_dict, annotated_frame)."""
        results = {
            "status":         "Focused",
            "face_count":     0,
            "no_face":        False,
            "multiple_faces": False,
            "head_left":      False,
            "head_right":     False,
            "drowsy":         False,
            "phone_detected": False,
            "unknown_face":   False,
            "ear":            None,
        }

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Haar face detection ──
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        results["face_count"] = len(faces)

        if len(faces) == 0:
            results["no_face"] = True
            results["status"]  = "No Face"
        elif len(faces) > 1:
            results["multiple_faces"] = True
            results["status"]         = "Multiple Faces"

        for (x, y, fw, fh) in faces:
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            if self.face_mesh is None:
                rel_x = (x + fw // 2) / w
                if rel_x < HEAD_LEFT_THRESHOLD:
                    results["head_left"] = True
                    results["status"]    = "Looking Left"
                elif rel_x > HEAD_RIGHT_THRESHOLD:
                    results["head_right"] = True
                    results["status"]     = "Looking Right"

            if _HAS_FACEREC and self.enrolled_encoding is not None:
                try:
                    box  = (y, x + fw, y + fh, x)
                    encs = face_recognition.face_encodings(rgb, [box])
                    if encs:
                        match = face_recognition.compare_faces(
                            [self.enrolled_encoding], encs[0],
                            tolerance=FACE_RECOGNITION_TOLERANCE,
                        )[0]
                        if not match:
                            results["unknown_face"] = True
                            results["status"]       = "Unknown Person"
                            cv2.putText(frame, "UNKNOWN", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, "VERIFIED", (x, y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                except Exception:
                    pass

        # ── Face Mesh: head-yaw + EAR drowsiness ──
        if self.face_mesh is not None:
            mesh = self.face_mesh.process(rgb)
            if mesh.multi_face_landmarks:
                lm = mesh.multi_face_landmarks[0].landmark

                nose_x  = lm[self.NOSE_TIP].x
                left_x  = lm[self.LEFT_CHEEK].x
                right_x = lm[self.RIGHT_CHEEK].x
                span    = right_x - left_x
                if span > 0.01:
                    rel_nose = (nose_x - left_x) / span
                    if rel_nose < HEAD_LEFT_THRESHOLD:
                        results["head_left"] = True
                        if results["status"] == "Focused":
                            results["status"] = "Looking Left"
                    elif rel_nose > HEAD_RIGHT_THRESHOLD:
                        results["head_right"] = True
                        if results["status"] == "Focused":
                            results["status"] = "Looking Right"

                left_ear  = self._ear(self.LEFT_EYE_IDX,  lm, w, h)
                right_ear = self._ear(self.RIGHT_EYE_IDX, lm, w, h)
                avg_ear   = (left_ear + right_ear) / 2.0
                results["ear"] = avg_ear

                if avg_ear < EAR_THRESHOLD:
                    self.ear_consec_count += 1
                else:
                    self.ear_consec_count = 0

                if self.ear_consec_count >= EAR_CONSEC_FRAMES:
                    results["drowsy"] = True
                    results["status"] = "Drowsiness Detected"
                    cv2.putText(frame, f"EAR {avg_ear:.2f} DROWSY", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, f"EAR {avg_ear:.2f}", (10, 65),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ── Phone detection via YOLO ──
        if self.yolo is not None:
            try:
                for r in self.yolo(frame, conf=YOLO_CONFIDENCE, verbose=False):
                    for box in r.boxes:
                        if self.yolo.names[int(box.cls)] == "cell phone":
                            results["phone_detected"] = True
                            results["status"]         = "Phone Detected"
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, "PHONE!", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception:
                pass

        color = (0, 255, 0) if results["status"] == "Focused" else (0, 165, 255)
        cv2.putText(frame, results["status"], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return results, frame