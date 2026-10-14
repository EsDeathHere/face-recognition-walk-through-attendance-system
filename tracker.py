# -*- coding: utf-8 -*-
"""Lightweight IoU tracker (ByteTrack-ish) for walk-through attendance."""

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class Track:
    def __init__(self, tid, det):
        self.id = tid
        self.bbox = list(det["bbox"])
        self.emb = det.get("emb")
        self.score = float(det.get("score", 0.0))
        self.sharpness = float(det.get("sharpness", 0.0))
        self.best_score = self.combined()
        self.best_emb = self.emb
        self.best_bbox = list(self.bbox)
        self.best_sharpness = self.sharpness
        self.missed = 0
        self.hits = 1
        self.age = 0

    def combined(self):
        return float(self.score) * max(1.0, self.sharpness / 50.0) if hasattr(self,'sharpness') else float(self.score)

    def update(self, det):
        self.bbox = list(det["bbox"])
        self.emb = det.get("emb", self.emb)
        self.score = float(det.get("score", self.score))
        self.sharpness = float(det.get("sharpness", self.sharpness))
        self.missed = 0
        self.hits += 1
        self.age += 1
        c = self.score * max(1.0, self.sharpness / 50.0)
        if c > self.best_score and self.emb is not None:
            self.best_score = c
            self.best_emb = self.emb
            self.best_bbox = list(self.bbox)
            self.best_sharpness = self.sharpness

    def to_dict(self):
        return {
            "id": self.id,
            "bbox": self.bbox,
            "emb": self.emb,
            "score": self.score,
            "sharpness": self.sharpness,
            "best_emb": self.best_emb,
            "best_score": self.best_score,
            "best_sharpness": self.best_sharpness,
            "hits": self.hits,
            "missed": self.missed,
        }


class SimpleTracker:
    def __init__(self, max_age=25, iou_thresh=0.3, min_hits=1):
        self.max_age = max_age
        self.iou_thresh = iou_thresh
        self.min_hits = min_hits
        self.tracks = []
        self.finished = []  # finalized tracks
        self._next_id = 1

    @staticmethod
    def iou(a, b):
        from utils import bbox_iou
        return bbox_iou(a, b)

    def update(self, detections):
        # detections: list of dicts
        # age tracks
        for t in self.tracks:
            t.age += 1

        if len(self.tracks) == 0:
            for d in detections:
                tr = Track(self._next_id, d)
                self._next_id += 1
                self.tracks.append(tr)
            return [t.to_dict() for t in self.tracks]

        # cost matrix
        cost = np.zeros((len(self.tracks), len(detections)), dtype=float)
        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                cost[i, j] = 1.0 - self.iou(t.bbox, d["bbox"])

        if _HAS_SCIPY and len(detections) > 0:
            r, c = linear_sum_assignment(cost)
        else:
            # greedy fallback
            r, c = self._greedy(cost)

        matched_t = set()
        matched_d = set()
        for i, j in zip(r, c):
            if i >= len(self.tracks) or j >= len(detections):
                continue
            if cost[i, j] <= (1.0 - self.iou_thresh):
                self.tracks[i].update(detections[j])
                matched_t.add(i)
                matched_d.add(j)

        # unmatched tracks -> increment missed, finalize if needed
        still = []
        for i, t in enumerate(self.tracks):
            if i in matched_t:
                still.append(t)
                continue
            t.missed += 1
            if t.missed >= self.max_age:
                if t.hits >= self.min_hits and t.best_emb is not None:
                    self.finished.append(t.to_dict())
            else:
                still.append(t)
        self.tracks = still

        # new tracks for unmatched detections
        for j, d in enumerate(detections):
            if j in matched_d:
                continue
            tr = Track(self._next_id, d)
            self._next_id += 1
            self.tracks.append(tr)

        return [t.to_dict() for t in self.tracks]

    def _greedy(self, cost):
        # returns arrays of matched indices
        rows = []
        cols = []
        used_r = set()
        used_c = set()
        # sort by cost
        flat = []
        for i in range(cost.shape[0]):
            for j in range(cost.shape[1]):
                flat.append((cost[i,j], i, j))
        flat.sort()
        for cv, i, j in flat:
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            rows.append(i); cols.append(j)
        return np.array(rows), np.array(cols)

    def pop_finished(self):
        out = self.finished[:]
        self.finished = []
        return out

