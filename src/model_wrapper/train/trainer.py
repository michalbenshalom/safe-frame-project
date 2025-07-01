import copy
from datetime import datetime
import os
from torch import nn, optim
import torch
from tqdm import tqdm

from src.config import MODEL_TYPE
def train(model, train_loader, val_loader, config, model_name):
    device = config.get("device", "cpu")
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    patience = config.get("early_stopping_patience", 3)
    save_dir = config.get("save_dir", ".")
    os.makedirs(save_dir, exist_ok=True)

    filename = generate_model_filename()
    save_path = os.path.join(save_dir, filename)

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_state, best_val_loss = None, float('inf')
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, epochs)
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch, epochs)

        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.2f}%, "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.2f}%")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, save_path)
            print(f"Saved best model so far (Val Loss={val_loss:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement. Patience: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                print("Early stopping.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model (Val Loss={best_val_loss:.4f})")

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(dataloader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}"):
        inputs, labels = preprocess(inputs, labels, device)

        optimizer.zero_grad()
        outputs = forward_pass(model, inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device, epoch, total_epochs):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"[Val] Epoch {epoch+1}/{total_epochs}"):
            inputs, labels = preprocess(inputs, labels, device)
            outputs = forward_pass(model, inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def preprocess(inputs, labels, device):
    inputs, labels = inputs.to(device), labels.to(device)

    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(0)
    if labels.ndim == 0:
        labels = labels.unsqueeze(0)
    labels = labels.long()

    return inputs, labels

def forward_pass(model, inputs):
    return model(inputs).logits if hasattr(model(inputs), 'logits') else model(inputs)

def generate_model_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{MODEL_TYPE}_{timestamp}_best.pt"