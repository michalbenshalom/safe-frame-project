import random
from sklearn.model_selection import train_test_split
from .image_dataset import ImageDataset

def split_dataset(image_paths, labels, val_size=0.005, test_size=0.99, random_seed=42):
    """
    Loads image paths & labels, shuffles, splits into train/val/test,
    and returns ImageDataset instances.
    """
    print(f"image_paths: {(image_paths)}")

    # Shuffle
    combined = list(zip(image_paths, labels))
    random.seed(random_seed)
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    # First split: Train vs Temp (Val+Test)
    val_test_size = val_size + test_size
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=val_test_size, random_state=random_seed
    )

    # Second split: Val vs Test
    val_ratio = val_size / val_test_size
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=1 - val_ratio, random_state=random_seed
    )

    # Create datasets
    train_dataset = ImageDataset(train_paths, train_labels)
    val_dataset = ImageDataset(val_paths, val_labels)
    test_dataset = ImageDataset(test_paths, test_labels)

    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset