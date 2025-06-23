from src.data_management.s3_manager import load_labels_for_category_s3, stream_s3_videos
from src.data_management.loader import video_to_frames
from src.data_management.validate import validate_video_file_single

import os
import cv2

def process_and_validate_videos():
    """
    Pipeline שמוריד קובצי וידאו מ-S3, טוען תיוגים לפי קטגוריה,
    מחלץ פריימים ושומר כל פריים לפי התיוג שלו (0 או 1).
    """
    all_results = []
    base_output_dir = "data/processed"
    labels_by_category = {}

    for category, event_folder, file_name, file_bytes in stream_s3_videos():

        # טען תיוגים לפי קטגוריה פעם אחת
        if category not in labels_by_category:
            labels_by_category[category] = load_labels_for_category_s3(category)

        labels = labels_by_category[category]
        key = (file_name, )  # כדי לחפש לפי שם קובץ ספציפי

        # ולידציה בסיסית לקובץ
        is_valid, msg = validate_video_file_single(file_name, file_bytes)
        if not is_valid:
            print(f"Skipping {category}/{event_folder}/{file_name}: {msg}")
            continue

        # חילוץ פריימים
        frames = video_to_frames(file_bytes, fps_interval=1)

        saved_count = 0
        for frame_idx, frame in frames:
            base_name = os.path.splitext(file_name)[0]
            label = labels.get((base_name), None)
            if label is None:
                continue  # אין תיוג לפריים הזה

            label_folder = "ONE" if label == 1 else "ZERO"
            save_dir = os.path.join(base_output_dir, label_folder, category, event_folder)
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{os.path.splitext(file_name)[0]}_frame_{frame_idx}.jpg")
            cv2.imwrite(save_path, frame)
            saved_count += 1

        print(f"{category}/{event_folder}/{file_name}: {saved_count} frames saved")
        all_results.append((category, event_folder, file_name, saved_count))

    return all_results