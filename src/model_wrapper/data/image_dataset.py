import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random
from config import CONFIG


def get_transforms(use_augmentation=False, is_train=True):
    """
    בונה את רשימת ה־transforms המתאימים לפי MODEL_TYPE ו־augmentation
    """
    # נרמול לפי סוג המודל
    if CONFIG["model_type"] == "resnet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    elif CONFIG["model_type"] == "vit":
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
    else:
        normalize = None

    transform_list = []

    if use_augmentation and is_train:
        # רק לאימון
        transform_list.extend([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        ])
    else:
        # ל־val/test ול־inference
        transform_list.append(transforms.Resize((224, 224)))

    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(normalize)

    return transforms.Compose(transform_list)


class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, use_augmentation=False, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = get_transforms(use_augmentation, is_train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


class InMemoryImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
        self.transform = get_transforms(use_augmentation=False, is_train=False)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = self.transform(image)
        return image
