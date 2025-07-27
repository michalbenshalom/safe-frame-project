import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from config import MODEL_TYPE

class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = self._build_transform(augment)

    def _build_transform(self, augment):
        if MODEL_TYPE == "resnet":
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif MODEL_TYPE == "vit":
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        else:
            normalize = None

        if augment:
            base_transforms = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]
        else:
            base_transforms = [
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ]

        if normalize:
            base_transforms.append(normalize)

        return transforms.Compose(base_transforms)

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
        self.transform = self._build_transform()

    def _build_transform(self):
        if MODEL_TYPE == "resnet":
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif MODEL_TYPE == "vit":
            normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        else:
            normalize = None

        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]

        if normalize:
            base_transforms.append(normalize)

        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")
        image = self.transform(image)
        return image
