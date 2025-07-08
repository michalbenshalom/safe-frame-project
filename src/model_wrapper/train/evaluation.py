from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int()

            all_preds.extend(preds.cpu().squeeze().tolist())
            all_labels.extend(labels.cpu().int().tolist())

    report = classification_report(all_labels, all_preds, output_dict=True)
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist(),
    }
