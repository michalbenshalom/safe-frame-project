import os
import random
from glob import glob
from sklearn.model_selection import train_test_split

from .image_dataset import ImageDataset

def load_dataset(dataset_path, test_size=0.3, random_seed=42):
    """
    Combines logic: loads images and labels from 'ONE' and 'ZERO',
    shuffles, splits to train/test, and returns ImageDataset instances.
    """
    print("Loading dataset...")

    image_paths = []
    labels = []

    for label_folder, label_value in [("ONE", 1), ("ZERO", 0)]:
        folder_path = os.path.join(dataset_path, label_folder)
        for img_path in glob(os.path.join(folder_path, "**", "*.jpg"), recursive=True):
            image_paths.append(img_path)
            labels.append(label_value)

    print(f"Found {len(image_paths)} images.")
    
    # Shuffle and split
    combined = list(zip(image_paths, labels))
    random.seed(random_seed)
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=test_size, random_state=random_seed
    )

    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels)
    test_dataset = ImageDataset(test_paths, test_labels)

    return train_dataset, test_dataset