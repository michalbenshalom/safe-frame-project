import copy
from datetime import datetime
import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_wrapper.models.base_model_wrapper import BaseModelWrapper
from src.config import CONFIG
from src.utils.s3_model_manager import S3ModelManager
from src.utils.logger import get_logger
from utils.ModelsTypes import MODEL_WRAPPERS

logger = get_logger()
s3_manager = S3ModelManager()  


def train(train_loader, val_loader, config, initial_model=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get("epochs", 5)
    lr = config.get("learning_rate", 2e-5)
    patience = config.get("early_stopping_patience", 3)
    log_dir = config.get("tensorboard_log_dir", "./runs")
    model_wrapper = MODEL_WRAPPERS[CONFIG["model_type"]]() 
    filename = model_wrapper.get_best_model_filename()
    s3_path = f"Models/{CONFIG['model_type']}/{filename}"

    if initial_model is not None:
        model_wrapper.model = initial_model
        logger.info("Continuing training with existing model")
    else:
        logger.info("Starting training with new model")

    model_wrapper.model.to(device)
    criterion = model_wrapper.criterion
    optimizer = optim.Adam(model_wrapper.model.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    # הדפסת פרטי האימון
    # הוסף הדפסה לבדיקת איזון הדאטה
    print(f"\n{'='*60}")
    print(f"Starting Training - Model: {CONFIG['model_type']}")
    print(f"Device: {device}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {config.get('batch_size', 32)}")
    print(f"Train Samples: {len(train_loader)}")
    print(f"Val Samples: {len(val_loader)}")
    print(f"{'='*60}\n")

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
        
        # הדפסות מפורטות של התקדמות האימון
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Epoch [{epoch+1}/{epochs}] - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 60)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)

        #michalbs : add to writer 
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

    # הדפסת סיכום האימון
    print(f"\n{'='*60}")
    print(f"Training Completed - Model: {CONFIG['model_type']}")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Total Epochs Trained: {len(history)}")
    if epochs_without_improvement >= patience:
        print(f"Training stopped early due to no improvement for {patience} epochs")
    print(f"{'='*60}\n")
    
    writer.close()
    #s3_manager.save_history(history, config.get("checkpoint_dir", "./checkpoints/"))//michalbs

    return {
        "model": model_wrapper.model,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_acc,
        "train_history": history
    }


def run_epoch(model_wrapper: BaseModelWrapper, dataloader, criterion, device, epoch, total_epochs,
              is_train: bool, optimizer=None):
    mode = "Train" if is_train else "Val"
    model_wrapper.model.train() if is_train else model_wrapper.model.eval()

    total_loss, correct, total = 0.0, 0, 0
    num_batches = len(dataloader)
    samples_processed = 0
    log_interval = 500  # הדפסה כל 500 דגימות

    with tqdm(dataloader, desc=f"[{mode}] Epoch {epoch+1}/{total_epochs}") as progress_bar:
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
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
            
            # עדכון מספר הדגימות המעובדות
            batch_size = inputs.size(0)
            samples_processed += batch_size

            # הדפסת loss כל 500 דגימות
            if samples_processed % log_interval < batch_size or batch_idx == num_batches - 1:
                current_loss = loss.detach().cpu().item()
                current_acc = (preds == labels).sum().item() / labels.numel() * 100 if labels.numel() > 0 else 0
                avg_loss_so_far = total_loss / (batch_idx + 1)
                print(f"[{mode}] Epoch {epoch+1}/{total_epochs} | Samples: {samples_processed} | "
                      f"Current Loss: {current_loss:.4f} | Avg Loss: {avg_loss_so_far:.4f} | "
                      f"Current Acc: {current_acc:.2f}%")

            # עדכון progress bar עם מידע מפורט יותר
            current_loss = loss.detach().cpu().item()
            current_acc = (preds == labels).sum().item() / labels.numel() * 100 if labels.numel() > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'acc': f'{current_acc:.2f}%',
                'batch': f'{batch_idx+1}/{num_batches}',
                'samples': f'{samples_processed}'
            })

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total * 100
    
    # הדפסת סיכום epoch
    print(f"[{mode}] Epoch {epoch+1}/{total_epochs} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Total Samples: {total}")
    
    return avg_loss, accuracy
