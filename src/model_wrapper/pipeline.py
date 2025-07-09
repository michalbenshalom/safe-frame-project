from loguru import logger
from .data.image_utils import load_dataset
from .data.dataset_loader import split_dataset
from src.config import CONFIG
from .train.trainer import train
from model_wrapper.models.vit_model import ViTModelWrapper
from model_wrapper.models.resnet_model import ResNetModelWrapper
from src.utils.s3_model_manager import S3ModelManager

s3_manager = S3ModelManager()

MODEL_REGISTRY = {
    "vit": lambda: ViTModelWrapper(CONFIG),
    "resnet": lambda: ResNetModelWrapper(CONFIG),
}

def run_models_pipeline(model_type , use_existing_model=False):
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
    
    model_wrapper = MODEL_REGISTRY[model_type]()    
    if use_existing_model:
        try:
            filename = model_wrapper.get_best_model_filename()
            s3_path = f"Models/{model_type}/{filename}"
            s3_manager.load_model(model_wrapper.model, s3_path)
            logger.info(f"Loaded model from s3://{s3_path}")
            return {"model": model_wrapper.model, "s3_path": s3_path, "loaded_from_s3": True}
        except Exception as e:
            logger.warning(f"Failed to load existing model. Training a new one. Error: {e}")
 

    image_paths, labels = load_dataset(CONFIG["dataset_path"])
    train_dataset, val_dataset, test_dataset = split_dataset(image_paths, labels)

    print(f"=== Training {model_type} ===")
    model_wrapper = MODEL_REGISTRY[model_type]() 

    trained_model = train(model_wrapper, train_dataset, val_dataset, CONFIG)
    return trained_model

