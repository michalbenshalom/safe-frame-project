import cv2
import os
import tempfile
import numpy as np

def video_to_frames(video_bytes, fps_interval=1):
    """
    מקבל קובץ וידאו כ־bytes, ומחזיר רשימת פריימים (תמונות) שנחולצו כל fps_interval שניות.
    """
    # שמירה זמנית של הקובץ כווידאו בדיסק
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        os.remove(tmp_path)
        return []

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_gap = int(fps * fps_interval) if fps > 0 else 25

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_gap == 0:
            frames.append((frame_idx, frame))

        frame_idx += 1

    cap.release()
    os.remove(tmp_path)
    return frames
