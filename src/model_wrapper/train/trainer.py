import copy
from datetime import datetime
import os
from torch import nn, optim
import torch
from tqdm import tqdm
import io
import boto3
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
from model_wrapper.models.base_model_wrapper import BaseModelWrapper
from src.config import MODEL_TYPE
from ..models.losses.loss_factory import get_loss_fn
from src.data_management.s3_manager import AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET_NAME
from src.utils.model_saver import save_best_model, save_train_history
from src.utils.logger import get_logger

load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
logger = get_logger()


def train(model_wrapper: BaseModelWrapper, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    patience = config.get("early_stopping_patience", 3)
    log_dir = config.get("tensorboard_log_dir", "./runs")
    filename = model_wrapper.generate_model_filename()
    s3_path = f"Models/{MODEL_TYPE}/{filename}"

    model_wrapper.model.to(device)
    loss_type = config.get("loss_type", "bce")
    loss_params = config.get("loss_params", {})
    criterion = get_loss_fn(loss_type, loss_params)
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    best_model_state, best_val_loss = None, float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model_wrapper, train_loader, optimizer, criterion, device, epoch, epochs)
        val_loss, val_acc = validate(model_wrapper, val_loader, criterion, device, epoch, epochs)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model_wrapper.model.state_dict())

            try:
                save_best_model(
                    model_state_dict=best_model_state,
                    bucket_name=AWS_BUCKET_NAME,
                    s3_path=s3_path,
                    aws_access_key=AWS_ACCESS_KEY,
                    aws_secret_key=AWS_SECRET_KEY
                )
                logger.info(f"Saved best model to s3://{AWS_BUCKET_NAME}/{s3_path} (Val Loss={val_loss:.4f})")
                epochs_without_improvement = 0
            except Exception as e:
                logger.error(f"Failed to save model to S3: {e}")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement. Patience: {epochs_without_improvement}/{patience}")
            if epochs_without_improvement >= patience:
                logger.warning("Early stopping.")
                break

    if best_model_state:
        model_wrapper.model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (Val Loss={best_val_loss:.4f})")

    writer.close()
    save_train_history(history, config=config)

    return {
        "model": model_wrapper.model,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "train_history": history,
        "s3_path": s3_path
    }


def train_one_epoch(model_wrapper: BaseModelWrapper, dataloader, optimizer, criterion, device, epoch, total_epochs):
    model_wrapper.model.train()
    total_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(dataloader, desc=f"[Train] Epoch {epoch+1}/{total_epochs}")

    for inputs, labels in progress_bar:
        inputs, labels = model_wrapper.preprocess(inputs, labels, device)
        optimizer.zero_grad()
        outputs = model_wrapper.forward_pass(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).long()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


def validate(model_wrapper: BaseModelWrapper, dataloader, criterion, device, epoch, total_epochs):
    
    model_wrapper.model.eval()
    total_loss, correct, total = 0.0, 0, 0
    progress_bar = tqdm(dataloader, desc=f"[Val] Epoch {epoch+1}/{total_epochs}")

    with torch.no_grad():
        for inputs, labels in progress_bar:
            inputs, labels = model_wrapper.preprocess(inputs, labels, device)
            outputs = model_wrapper.forward_pass(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            progress_bar.set_postfix(batch_loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy


