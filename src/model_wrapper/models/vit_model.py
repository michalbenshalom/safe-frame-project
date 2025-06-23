import torch
from torch import nn
from torchvision.models import vit_b_16
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from src.model_wrapper.data.dataset_loader import load_dataset
from src.model_wrapper.data.image_dataset import ImageDataset
from src.model_wrapper.train.trainer import train_model
from src.model_wrapper.train.evaluation import evaluate_model

def trainModel():
    image_paths, labels = load_dataset()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(image_paths, labels, transform=transform)

    # פיצול ל-train/val/test
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4)
    test_loader = DataLoader(test_ds, batch_size=4)

    # === הגדרת ViT ===
    model = vit_b_16(pretrained=True)
    model.heads.head = nn.Linear(model.heads.head.in_features, 1)  # התאמה לבעיה בינארית

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = train_model(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), "vit_b16_trained.pth")
    print("✅ Model saved to vit_b16_trained.pth")

    eval_result = evaluate_model(model, test_loader, device)

    return {
        "model": "ViT",
        "status": "trained_and_evaluated",
        "saved_as": "vit_b16_trained.pth",
        "accuracy": eval_result["accuracy"],
        "precision": eval_result["precision"],
        "recall": eval_result["recall"],
        "f1_score": eval_result["f1_score"]
    }