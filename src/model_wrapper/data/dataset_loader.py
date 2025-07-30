import random
from sklearn.model_selection import train_test_split
from collections import Counter
from config import CONFIG
from .image_dataset import ImageDataset

def print_label_distribution(name, labels):
    counter = Counter(labels)
    total = len(labels)
    print(f"\n{name} label distribution:")
    for label, count in counter.items():
        percent = 100 * count / total
        print(f"  Label {label}: {count} ({percent:.1f}%)")

def split_dataset(image_paths, labels, random_seed=42):
    """
    Shuffle and split dataset into train/val/test with stratification.
    Returns ImageDataset instances for each split.
    """
    combined = list(zip(image_paths, labels))
    random.seed(random_seed)
    random.shuffle(combined)

    # 💡 שלב בחירה בין limit ל-percent
    if "limit_dataset" in CONFIG and CONFIG["limit_dataset"] is not None:
        combined = combined[:min(CONFIG["limit_dataset"], len(combined))]
        print(f"\n🔹 Using fixed limit of {len(combined)} samples")
    else:
        percent = CONFIG.get("dataset_percent", 100)
        limit = int(len(combined) * percent / 100)
        combined = combined[:limit]
        print(f"\n🔹 Using {percent}% of data: {len(combined)} samples")

    image_paths, labels = zip(*combined)

    val_size = CONFIG.get("val_size", 0.1)
    test_size = CONFIG.get("test_size", 0.1)
    val_test_size = val_size + test_size

    # Split: train / val+test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels,
        test_size=val_test_size,
        stratify=labels,
        random_state=random_seed
    )

    # Split: val / test
    val_ratio = val_size / val_test_size
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=1 - val_ratio,
        stratify=temp_labels,
        random_state=random_seed
    )

    # 🔁 Augmentation toggle
    use_augmentation = CONFIG.get("use_augmentation", False)

    # 📦 Create dataset objects
    train_dataset = ImageDataset(train_paths, train_labels, use_augmentation=use_augmentation, is_train=True)
    val_dataset = ImageDataset(val_paths, val_labels, use_augmentation=False, is_train=False)
    test_dataset = ImageDataset(test_paths, test_labels, use_augmentation=False, is_train=False)

    # 📊 Print dataset sizes
    print(f"\n📊 Dataset Sizes:")
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    if use_augmentation:
        print("✅ Using data augmentation for training")

    # 🧮 Print label distribution
    print_label_distribution("Train", train_labels)
    print_label_distribution("Validation", val_labels)
    print_label_distribution("Test", test_labels)

    return train_dataset, val_dataset, test_dataset
