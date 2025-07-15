import copy
from datetime import datetime
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_wrapper.models.base_model_wrapper import BaseModelWrapper
from config import MODEL_TYPE
from utils.s3_model_manager import S3ModelManager
from utils.logger import get_logger

logger = get_logger()
s3_manager = S3ModelManager()  


def train(model_wrapper: BaseModelWrapper, train_loader, val_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    patience = config.get("early_stopping_patience", 3)
    log_dir = config.get("tensorboard_log_dir", "./runs")
    filename = model_wrapper.get_best_model_filename()
    s3_path = f"Models/{MODEL_TYPE}/{filename}"

    model_wrapper.model.to(device)
    criterion = model_wrapper.criterion
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    best_model_state, best_val_loss = None, float('inf')
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model_wrapper, train_loader, criterion, device, epoch, epochs, is_train=True, optimizer=optimizer
        )

        val_loss, val_acc = run_epoch(
            model_wrapper, val_loader, criterion, device, epoch, epochs, is_train=False
        )
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
                #s3_manager.save_model(best_model_state, s3_path)//michalbs
                logger.info(f"Saved best model to s3://{s3_manager.bucket_name}/{s3_path} (Val Loss={val_loss:.4f})")
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
    #s3_manager.save_history(history, config.get("checkpoint_dir", "./checkpoints/"))//michalbs

    return {
        "model": model_wrapper.model,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "train_history": history,
        "s3_path": s3_path
    }


def run_epoch(model_wrapper: BaseModelWrapper, dataloader, criterion, device, epoch, total_epochs,
              is_train: bool, optimizer=None):
    mode = "Train" if is_train else "Val"
    model_wrapper.model.train() if is_train else model_wrapper.model.eval()

    total_loss, correct, total = 0.0, 0, 0

    with tqdm(dataloader, desc=f"[{mode}] Epoch {epoch+1}/{total_epochs}") as progress_bar:
        for inputs, labels in progress_bar:
            inputs, labels = model_wrapper.preprocess(inputs, labels, device)

            if is_train:
                optimizer.zero_grad()

            with torch.set_grad_enabled(is_train):
                outputs = model_wrapper.forward_pass(inputs)
                loss = criterion(outputs, labels)

                if is_train:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item()
            preds = model_wrapper.predict(outputs)
            correct += (preds == labels).sum().item()
            total += labels.numel() if preds.shape == labels.shape else preds.shape[0]

            progress_bar.set_postfix(batch_loss=loss.detach().cpu().item())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    return avg_loss, accuracy
