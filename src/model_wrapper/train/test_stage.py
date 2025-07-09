
import torch
from torch.utils.data import DataLoader
from src.model_wrapper.data.dataset_loader import split_dataset
from src.model_wrapper.train.evaluation import evaluate_model
from src.config import MODEL_TYPE, CONFIG
from model_wrapper.models.vit_model import ViTModelWrapper
from model_wrapper.models.resnet_model import ResNetModelWrapper

MODEL_WRAPPERS = {
    "vit": ViTModelWrapper,
    "resnet": ResNetModelWrapper,
}

def test_model(model_type=None, weights_path=None):
    model_type = (model_type or MODEL_TYPE).lower()
    if model_type not in MODEL_WRAPPERS:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load data
    _, _, test_dataset = split_dataset(*load_dataset(CONFIG["dataset_path"]))
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False)

    # Load model
    model_wrapper = MODEL_WRAPPERS[model_type](CONFIG)
    model = model_wrapper.model
    if weights_path is None:
        weights_path = f"{model_type}_trained.pth"
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Evaluate
    eval_result = evaluate_model(model, test_loader, device)
    return eval_result

# Helper to load dataset
from src.model_wrapper.data.image_utils import load_dataset
