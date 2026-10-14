# -*- coding: utf-8 -*-
"""Auto-Absen Walk-Through: detect, track, recognize, log attendance."""

import time
import threading
from collections import deque

import cv2
import numpy as np

from config import (
    CONF_RUNTIME, CAMERA_INDEX, DISPLAY, MATCH_THRESHOLD, ATTENDANCE_COOLDOWN_SEC,
    BEST_FRAME_MIN_SHARPNESS,
)
from detector import FaceDetector
from tracker import SimpleTracker
from database import init_db, get_all_profiles
from recognizer import match_embedding, cooldown_ok
from utils import laplacian_variance, bbox_iou


class AsyncVideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.queue = deque(maxlen=1)
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            with self.lock:
                self.queue.append(frame)

    def read(self):
        with self.lock:
            if not self.queue:
                return None
            return self.queue[-1].copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=2)
        self.cap.release()


def process_finished_track(track, gallery):
    best_emb = track.get("best_emb")
    if best_emb is None:
        return
    # sharpness gate
    bs = track.get("best_sharpness", 0.0)
    if bs < BEST_FRAME_MIN_SHARPNESS:
        # still try if detection score decent
        pass
    uid, name, score = match_embedding(best_emb, gallery)
    if uid is None:
        print(f"[Track {track['id']}] No match (best={score:.3f})")
        return
    if not cooldown_ok(uid):
        print(f"[Track {track['id']}] {name} already attended recently (cooldown)")
        return
    from database import log_attendance, can_attend
    if not can_attend(uid):
        return
    log_attendance(uid, name, score)
    print(f"[ATTENDANCE] {name} (score={score:.3f}) id={uid}")


def main():
    init_db()
    gallery = get_all_profiles()
    print(f"Loaded {len(gallery)} embeddings from DB.")

    det = FaceDetector(mode="runtime")
    tracker = SimpleTracker()
    stream = AsyncVideoStream(CAMERA_INDEX)

    fps = 0.0
    t0 = time.time()
    frame_count = 0
    last_gallery_refresh = time.time()

    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.03)
            continue

        frame_count += 1
        # periodic gallery refresh (every 30s) to pick up new registrations
        if time.time() - last_gallery_refresh > 30:
            gallery = get_all_profiles()
            last_gallery_refresh = time.time()

        # detect
        faces = det.detect(frame)
        dets = []
        for f in faces:
            bbox = det.bbox(f)
            emb = det.emb(f)
            if emb is None:
                continue
            sharp = det.blur_score(frame, f)
            # quality gate: skip very blurry
            if sharp < 10:
                continue
            dets.append({
                "bbox": bbox,
                "emb": emb,
                "score": det.score(f),
                "sharpness": sharp,
            })

        tracks = tracker.update(dets)
        finished = tracker.pop_finished()
        for tr in finished:
            process_finished_track(tr, gallery)

        # draw
        for tr in tracks:
            x1, y1, x2, y2 = [int(v) for v in tr["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"ID {tr['id']} sharp={tr.get('sharpness',0):.0f}"
            cv2.putText(frame, label, (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # fps
        if frame_count % 5 == 0:
            dt = time.time() - t0
            fps = frame_count / dt if dt > 0 else 0.0

        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(frame, f"Gallery: {len(gallery)} emb", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(frame, "q quit | r reload gallery", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        if DISPLAY:
            cv2.imshow("Auto-Absen", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('r'):
            gallery = get_all_profiles()
            last_gallery_refresh = time.time()
            print(f"Gallery reloaded: {len(gallery)} embeddings")

    stream.stop()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()

