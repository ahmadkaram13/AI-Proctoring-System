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

# Detection thresholds
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

# Paths & camera
SCREENSHOT_DIR = "screenshots"
REPORTS_DIR    = "reports"
RECORDINGS_DIR = "recordings"
YOLO_MODEL     = "yolov8n.pt"
CAMERA_INDEX   = 0

# GUI
FEED_UPDATE_MS = 30
WINDOW_WIDTH   = 920
WINDOW_HEIGHT  = 620

# Weighted penalties — every event type has its own cost
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
        """Log an event if cooldown has passed. Returns True if logged."""
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
        """Export events to a timestamped Excel file."""
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