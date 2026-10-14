# -*- coding: utf-8 -*-
"""Interactive multi-angle face enrollment."""

import time
import cv2
import numpy as np

from config import (
    TARGET_EMBEDDINGS, DIVERSITY_COSINE, CAPTURE_COOLDOWN_SEC, BLUR_THRESHOLD,
    MIN_FACE_SIZE, CAMERA_INDEX, DISPLAY,
)
from detector import FaceDetector
from database import init_db, add_user, add_embedding, user_embedding_count
from utils import laplacian_variance, face_crop


def main():
    init_db()
    name = input("Enter user name: ").strip()
    if not name:
        print("Name required")
        return
    uid = add_user(name)
    print(f"User created id={uid}. Look at camera and move head slowly...")

    det = FaceDetector(mode="enroll")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    captured = []  # list of (emb, blur, yaw/pitch)
    last_capture = 0.0
    running = True
    while running:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.flip(frame, 1)
        faces = det.detect(frame)
        face = det.largest(faces)

        status = f"Captured {len(captured)}/{TARGET_EMBEDDINGS}"
        if face is not None:
            bbox = det.bbox(face)
            x1, y1, x2, y2 = bbox
            blur = det.blur_score(frame, face)
            yaw, pitch, roll = getattr(face, "_yaw", 0), getattr(face, "_pitch", 0), getattr(face, "_roll", 0)
            # size check
            size_ok = (x2-x1) >= MIN_FACE_SIZE and (y2-y1) >= MIN_FACE_SIZE
            sharp_ok = blur >= BLUR_THRESHOLD
            color = (0, 255, 0) if (size_ok and sharp_ok) else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"blur={blur:.1f} yaw={yaw:.2f} pitch={pitch:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            # attempt capture
            now = time.time()
            if size_ok and sharp_ok and (now - last_capture) >= CAPTURE_COOLDOWN_SEC:
                emb = det.emb(face)
                if emb is not None:
                    # diversity check
                    diverse = True
                    for c_emb, *_ in captured:
                        from utils import cosine_similarity
                        if cosine_similarity(emb, c_emb) > DIVERSITY_COSINE:
                            diverse = False
                            break
                    if diverse or len(captured) == 0:
                        captured.append((emb, blur, (yaw, pitch, roll)))
                        add_embedding(uid, emb, angle=f"pose_{len(captured)}", score=blur)
                        last_capture = now
                        print(f"  captured {len(captured)} (blur={blur:.1f})")
            guide = "Move head: front / left / right / up / down"
        else:
            guide = "No face detected - face the camera"
            color = (0, 0, 255)

        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, guide, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # progress bar
        prog = int((len(captured) / TARGET_EMBEDDINGS) * 300)
        cv2.rectangle(frame, (10, 80), (10 + prog, 95), (0, 255, 0), -1)
        cv2.rectangle(frame, (10, 80), (310, 95), (255, 255, 255), 1)
        cv2.imshow("Enroll", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        if key == ord('s'):
            # force save current if face good
            if face is not None and det.emb(face) is not None:
                emb = det.emb(face)
                blur = det.blur_score(frame, face)
                captured.append((emb, blur, (0, 0, 0)))
                add_embedding(uid, emb, angle=f"manual_{len(captured)}", score=blur)
                print(f"  forced captured {len(captured)}")
        if len(captured) >= TARGET_EMBEDDINGS:
            print("Target reached.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. {len(captured)} embeddings saved for user {name} (id={uid}).")
    print(f"Total in DB: {user_embedding_count(uid)}")


if __name__ == "__main__":
    main()

