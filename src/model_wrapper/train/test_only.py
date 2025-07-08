# # train/test_only.py

# import torch
# from torchvision import models, transforms
# from torch import nn
# from torch.utils.data import DataLoader, random_split
# from src.model_wrapper.data.dataset_loader import split_dataset
# from src.model_wrapper.data.image_dataset import ImageDataset
# from src.model_wrapper.train.evaluation import evaluate_model
# from src.config import MODEL_TYPE

# def load_model(model_type: str, weights_path: str):
#     if model_type.lower() == "resnet":
#         model = models.resnet18(pretrained=False)
#         model.fc = nn.Linear(model.fc.in_features, 1)
#     elif model_type.lower() == "vit":
#         model = models.vit_b_16(pretrained=False)
#         model.heads.head = nn.Linear(model.heads.head.in_features, 1)
#     else:
#         raise ValueError(f"Unknown model type: {model_type}")
    
#     model.load_state_dict(torch.load(weights_path, map_location="cpu"))
#     return model

# def test_model(model_type="resnet", weights_path="resnet18_trained.pth"):
#     image_paths, labels = split_dataset()
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),
#         transforms.ToTensor()
#     ])
#     dataset = ImageDataset(image_paths, labels, transform=transform)

#     train_size = int(0.7 * len(dataset))
#     val_size = int(0.15 * len(dataset))
#     test_size = len(dataset) - train_size - val_size
#     _, _, test_ds = random_split(dataset, [train_size, val_size, test_size])
#     test_loader = DataLoader(test_ds, batch_size=32)

#     model = load_model(model_type, weights_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     eval_result = evaluate_model(model, test_loader, device)
#     return eval_result
