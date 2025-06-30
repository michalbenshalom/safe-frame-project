from src.config import CONFIG
from .data.dataset_loader import load_dataset
from .models.resnet_model import get_resnet_model as trainResNet
from .models.vit_model import trainModel as trainViT
from .train.trainer import train

MODEL_REGISTRY = {
    "vit": trainViT,
    "resnet": trainResNet,
    # "blip": get_blip_model,
    # "llava": get_llava_model
}

def run_models_pipeline(model_type):
    """
    Run the pipeline based on the selected model type.
    Args: model_type (str): The type of model to train ("ViT" or "ResNet").
    Returns: dict: Training/evaluation metrics or None if invalid type.
    """
    train_dataset, test_dataset = load_dataset(CONFIG["dataset_path"], CONFIG["batch_size"])
    print(f"=== Training {model_type} ===")
    model_fn = MODEL_REGISTRY[model_type.lower()]
    hf_name = CONFIG["model_hf_names"][model_type.lower()]
    model = model_fn(CONFIG["num_classes"], hf_name)
    train(model, train_dataset, test_dataset, CONFIG, model_type.lower())
    return {
        "model_type": model_type,
        "num_classes": CONFIG["num_classes"],
        "batch_size": CONFIG["batch_size"],
        "dataset_path": CONFIG["dataset_path"]
    }

