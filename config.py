# -*- coding: utf-8 -*-
"""Configuration for Auto-Absen Face Recognition Walk-Through."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "attendance.db")

DETECTOR_MODEL = os.getenv("DETECTOR_MODEL", "buffalo_l")
DET_SIZE_ENROLL = (320, 320)
DET_SIZE_RUNTIME = (320, 320)

CONF_ENROLL = float(os.getenv("CONF_ENROLL", "0.35"))
CONF_RUNTIME = float(os.getenv("CONF_RUNTIME", "0.5"))

BLUR_THRESHOLD = float(os.getenv("BLUR_THRESHOLD", "40.0"))
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "60"))

TARGET_EMBEDDINGS = int(os.getenv("TARGET_EMBEDDINGS", "8"))
DIVERSITY_COSINE = float(os.getenv("DIVERSITY_COSINE", "0.88"))
CAPTURE_COOLDOWN_SEC = float(os.getenv("CAPTURE_COOLDOWN_SEC", "0.4"))

MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.45"))
ATTENDANCE_COOLDOWN_SEC = int(os.getenv("ATTENDANCE_COOLDOWN_SEC", "300"))

TRACK_MAX_AGE = int(os.getenv("TRACK_MAX_AGE", "25"))
TRACK_IOU_THRESH = float(os.getenv("TRACK_IOU_THRESH", "0.3"))
BEST_FRAME_MIN_SHARPNESS = float(os.getenv("BEST_FRAME_MIN_SHARPNESS", "30.0"))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
DISPLAY = os.getenv("DISPLAY_WINDOW", "1") == "1"
CONTEXT_ID = int(os.getenv("CONTEXT_ID", "-1"))  # -1 CPU, 0 GPU
