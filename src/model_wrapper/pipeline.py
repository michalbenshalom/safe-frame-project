from .data.image_utils import load_dataset
from .data.dataset_loader import split_dataset
from src.config import CONFIG
from .train.trainer import train
from model_wrapper.models.vit_model import ViTModelWrapper
from model_wrapper.models.resnet_model import ResNetModelWrapper

MODEL_REGISTRY = {
    "vit": lambda: ViTModelWrapper(CONFIG),
    "resnet": lambda: ResNetModelWrapper(CONFIG),
}

def run_models_pipeline(model_type):
    """
    Run the pipeline based on the selected model type.
    Args: 
        model_type (str): The type of model to train ("ViT" or "ResNet").
    Returns: 
        dict: Training/evaluation metrics or None if invalid type.
    """
    model_type = model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    image_paths, labels = load_dataset(CONFIG["dataset_path"])
    train_dataset, val_dataset, test_dataset = split_dataset(image_paths, labels)

    print(f"=== Training {model_type} ===")
    model_wrapper = MODEL_REGISTRY[model_type]() 

    result = train(model_wrapper, train_dataset, val_dataset, CONFIG)
    return result

