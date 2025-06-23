# dataset_loader.py

import os
from glob import glob

def load_dataset():
    """
    טוען את כל התמונות והתיוגים מהתיקיות data/processed/ONE ו-ZERO.
    מחזיר שתי רשימות: paths, labels
    """
    print("Loading dataset...")
    data_dir = os.path.join("data", "processed")
    image_paths = []
    labels = []
    for label_folder, label_value in [("ONE", 1), ("ZERO", 0)]:
        folder_path = os.path.join(data_dir, label_folder)
        # חפש את כל קבצי ה-jpg בכל תתי התיקיות
        for img_path in glob(os.path.join(folder_path, "*", "*", "*.jpg")):
            image_paths.append(img_path)
            labels.append(label_value)
    print(f"Loaded {len(image_paths)} images.")
    return image_paths, labels
