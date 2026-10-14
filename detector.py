# -*- coding: utf-8 -*-
"""Face detector/recognition wrapper (InsightFace SCRFD + ArcFace)."""

import os
import cv2
import numpy as np

from config import DETECTOR_MODEL, DET_SIZE_ENROLL, DET_SIZE_RUNTIME, CONF_ENROLL, CONF_RUNTIME, CONTEXT_ID
from utils import estimate_pose, laplacian_variance, normalize_emb, face_crop

try:
    import insightface
    from insightface.app import FaceAnalysis
    _HAS_INSIGHT = True
except Exception as e:
    _HAS_INSIGHT = False
    _INSIGHT_ERR = e


class FaceDetector:
    def __init__(self, mode="runtime", det_size=None, conf=None):
        if not _HAS_INSIGHT:
            raise RuntimeError(f"insightface not available: {_INSIGHT_ERR}")
        self.mode = mode
        self.model_name = DETECTOR_MODEL
        if det_size is None:
            det_size = DET_SIZE_ENROLL if mode == "enroll" else DET_SIZE_RUNTIME
        if conf is None:
            conf = CONF_ENROLL if mode == "enroll" else CONF_RUNTIME
        self.conf = float(conf)
        self.det_size = det_size
        providers = ["CPUExecutionProvider"]
        if CONTEXT_ID >= 0:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.app = FaceAnalysis(name=self.model_name, providers=providers)
        # det_thresh low; we filter manually for flexibility
        self.app.prepare(ctx_id=CONTEXT_ID, det_size=self.det_size, det_thresh=0.1)
        # recognition model threshold not used; we compute manually

    def detect(self, frame, conf=None):
        if conf is None:
            conf = self.conf
        if frame is None:
            return []
        try:
            faces = self.app.get(frame)
        except Exception:
            return []
        out = []
        for f in faces:
            ds = float(getattr(f, "det_score", 1.0))
            if ds < conf:
                continue
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            # min size filter
            if (bbox[2]-bbox[0]) < 20 or (bbox[3]-bbox[1]) < 20:
                continue
            # attach pose estimate
            kps = getattr(f, "kps", None)
            yaw, pitch, roll = (0.0, 0.0, 0.0)
            if kps is not None:
                yaw, pitch, roll = estimate_pose(kps)
            f._yaw = yaw
            f._pitch = pitch
            f._roll = roll
            if getattr(f, "embedding", None) is not None:
                f.embedding = normalize_emb(f.embedding)
            out.append(f)
        return out

    def largest(self, faces):
        if not faces:
            return None
        def area(f):
            b = f.bbox
            return (b[2]-b[0])*(b[3]-b[1])
        return max(faces, key=area)

    @staticmethod
    def bbox(f):
        return [int(v) for v in f.bbox]

    @staticmethod
    def kps(f):
        return getattr(f, "kps", None)

    @staticmethod
    def emb(f):
        e = getattr(f, "embedding", None)
        if e is None:
            return None
        return normalize_emb(e)

    @staticmethod
    def score(f):
        return float(getattr(f, "det_score", 0.0))

    def blur_score(self, frame, f):
        crop = face_crop(frame, f.bbox)
        if crop is None or crop.size == 0:
            return 0.0
        return laplacian_variance(crop)

