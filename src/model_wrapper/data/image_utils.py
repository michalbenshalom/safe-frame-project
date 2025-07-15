import os
from glob import glob

def load_dataset(dataset_path):
    """
    Scans the dataset directory and returns image paths and binary labels (0/1).
    Assumes folders named 'ONE' (label=1) and 'ZERO' (label=0).
    """
    image_paths = []
    labels = []

    for label_folder, label_value in [("ONE", 1), ("ZERO", 0)]:
        folder_path = os.path.join(dataset_path, label_folder)
        for img_path in glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True):
            image_paths.append(img_path)
            labels.append(label_value)

    return image_paths, labels