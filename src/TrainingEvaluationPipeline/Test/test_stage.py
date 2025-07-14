
import torch
from torch.utils.data import DataLoader
from TrainingEvaluationPipeline.Evaluation.evaluation import evaluate_model
from src.config import MODEL_TYPE, CONFIG
from utils.ModelsTypes import MODEL_WRAPPERS


def test_model(model, test_dataset=None, weights_path=None):
    if MODEL_TYPE not in MODEL_WRAPPERS:
        raise ValueError(f"Unknown model type: {MODEL_TYPE}")

    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_wrapper = MODEL_WRAPPERS[MODEL_TYPE]()  # יצירת מופע בפועל
    model_wrapper.model = model

    eval_result = evaluate_model(model_wrapper, test_loader, device)
    return eval_result


