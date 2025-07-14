from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch
import pandas as pd
import os
from utils.logger import get_logger

logger = get_logger()

def evaluate_model(model_wrapper, test_loader, device, save_csv_path: str = None):
    model_wrapper.model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        print(f"Evaluating batch with {len(test_loader)} images")
        i = 0
        for images, labels in test_loader:
            i = i+1
            print(f"Evaluating batch index {i} , images = {len(images)}")
            images, labels = images.to(device), labels.to(device)
            outputs = model_wrapper.forward_pass(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            all_preds.extend(preds.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    report_dict = classification_report(all_labels, all_preds, output_dict=True)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # DataFrame מסודר
    df = pd.DataFrame(report_dict).T
    summary_df = df[['precision', 'recall', 'f1-score']].round(3)

    # הדפסה
    print("\n Classification Report:")
    print(summary_df)
    print("\n Accuracy:", round(acc * 100, 2), "%")
    print("\n Confusion Matrix:")
    cm_df = pd.DataFrame(cm, index=["True: 0", "True: 1"], columns=["Pred: 0", "Pred: 1"])
    print(cm_df)

    # כתיבה ללוג
    logger.info("\n" + summary_df.to_string())
    logger.info(f"Accuracy: {acc:.4f}")
    logger.info("\nConfusion Matrix:\n" + cm_df.to_string())

    return {
        "accuracy": acc,
        "precision": report_dict["1"]["precision"],
        "recall": report_dict["1"]["recall"],
        "f1_score": report_dict["1"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict
    }
