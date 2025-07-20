from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import pandas as pd
import numpy as np
import os
from utils.logger import get_logger

logger = get_logger()

def evaluate_model(model_wrapper, test_loader, device, save_csv_path: str = None):
    model_wrapper.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        print(f"Evaluating batch with {len(test_loader)} batches")
        for i, (images, labels) in enumerate(test_loader, start=1):
            print(f"Evaluating batch index {i} , images = {len(images)}")
            images, labels = images.to(device), labels.to(device)
            outputs = model_wrapper.forward_pass(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            all_preds.extend(preds.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    unique_labels = np.unique(np.concatenate([all_labels, all_preds]))
    missing_labels = set([0, 1]) - set(unique_labels)
    if missing_labels:
        print(f"⚠️ Warning: Missing labels in evaluation: {missing_labels}")

    # קיבוע labelים גם אם לא הופיעו בפועל
    labels = [0, 1]
    cm = confusion_matrix(all_labels, all_preds, labels=labels)

    report_dict = classification_report(all_labels, all_preds, output_dict=True, labels=labels, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    # דיווח
    df = pd.DataFrame(report_dict).T
    summary_df = df[['precision', 'recall', 'f1-score']].round(3)

    print("\n Classification Report:")
    print(summary_df)
    print("\n Accuracy:", round(acc * 100, 2), "%")
    print("\n Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])
    print(cm_df)

    # לוגים
    logger.info("\n" + summary_df.to_string())
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\nConfusion Matrix:\n" + cm_df.to_string())

    return {
        "accuracy": acc,
        "precision": report_dict.get("1", {}).get("precision", 0.0),
        "recall": report_dict.get("1", {}).get("recall", 0.0),
        "f1_score": report_dict.get("1", {}).get("f1-score", 0.0),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
