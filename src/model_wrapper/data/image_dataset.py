import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import random

def get_transforms(use_augmentation=False, is_train=True):
    """
    מחזיר transforms מתאימים - עם או בלי augmentation
    """
    if use_augmentation and is_train:
        # Transforms עם augmentation לאימון
        return transforms.Compose([
            transforms.Resize((256, 256)),  # קצת יותר גדול
            transforms.RandomCrop((224, 224)),  # חיתוך אקראי
            transforms.RandomHorizontalFlip(p=0.5),  # היפוך אופקי
            transforms.RandomRotation(degrees=10),  # סיבוב קל
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # שינוי צבעים
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # נרמול
        ])
    else:
        # Transforms בסיסיים לבדיקה/validation
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, use_augmentation=False, is_train=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or get_transforms(use_augmentation, is_train)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)        
        return image, label

class InMemoryImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images  # list of numpy arrays or PIL Images
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert numpy array to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = self.transform(image)
        return image


