# -*- coding: utf-8 -*-
"""Utility helpers for face processing, quality, pose, embeddings."""

import cv2
import numpy as np


def laplacian_variance(gray_or_bgr):
    """Compute blur score (higher = sharper)."""
    if gray_or_bgr is None:
        return 0.0
    if len(gray_or_bgr.shape) == 3:
        gray = cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_or_bgr
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_crop(frame, bbox, margin=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, x1 - int(bw * margin))
    y1 = max(0, y1 - int(bh * margin))
    x2 = min(w, x2 + int(bw * margin))
    y2 = min(h, y2 + int(bh * margin))
    return frame[y1:y2, x1:x2]


def estimate_pose(kps):
    """Estimate yaw/pitch/roll-ish ratios from 5-point landmarks."""
    try:
        le, re, nose, lm, rm = [np.array(p, dtype=float) for p in kps]
    except Exception:
        return 0.0, 0.0, 0.0
    eye_center = (le + re) / 2.0
    eye_dist = float(np.linalg.norm(re - le)) + 1e-6
    yaw = float((nose[0] - eye_center[0]) / eye_dist)
    pitch = float((nose[1] - eye_center[1]) / eye_dist)
    roll = float(np.degrees(np.arctan2(re[1] - le[1], re[0] - le[0])))
    return yaw, pitch, roll


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize_emb(emb):
    emb = np.asarray(emb, dtype=np.float32).ravel()
    n = np.linalg.norm(emb)
    if n > 0:
        emb = emb / n
    return emb


def emb_to_bytes(emb):
    return np.asarray(emb, dtype=np.float32).tobytes()


def bytes_to_emb(b):
    return np.frombuffer(b, dtype=np.float32).copy()


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def draw_text(frame, text, org, color=(0, 255, 0), scale=0.6, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def put_fps(frame, fps):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


def largest_face(faces):
    if not faces:
        return None
    def area(f):
        b = f.bbox if hasattr(f, "bbox") else f["bbox"]
        return (b[2]-b[0])*(b[3]-b[1])
    return max(faces, key=area)

