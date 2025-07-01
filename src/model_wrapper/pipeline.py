from .data.image_utils import load_dataset
from .data.dataset_loader import split_dataset
from src.config import CONFIG
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
    image_paths, labels = load_dataset(CONFIG["dataset_path"])
    train_dataset, val_dataset, test_dataset = split_dataset(image_paths, labels)
    print(f"=== Training {model_type} ===")
    model_fn = MODEL_REGISTRY[model_type.lower()] 
    model = model_fn()
    train(model, train_dataset, val_dataset, CONFIG, model_type.lower())
    return {
        "model_type": model_type,
        "num_classes": CONFIG["num_classes"],
        "batch_size": CONFIG["batch_size"],
        "dataset_path": CONFIG["dataset_path"]
    }

