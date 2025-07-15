from loguru import logger
from training_evaluation.Test.test_stage import test_model
from model_wrapper.data.image_utils import load_dataset
from model_wrapper.data.dataset_loader import split_dataset
from config import CONFIG, MODEL_TYPE, USE_EXISTING_MODEL
from utils.ModelsTypes import MODEL_WRAPPERS
from training_evaluation.train.trainer import train
from utils.s3_model_manager import S3ModelManager

s3_manager = S3ModelManager()


def run_models_pipeline():
    """
    Runs the full pipeline, returning test dataset and the trained or loaded model.
    Returns:
        Tuple[test_dataset, model]: Evaluation dataset and the model instance.
    """
    test_dataset, trained_model = get_or_train_model() 
    return test_model(trained_model, test_dataset)

def get_or_train_model():
    """
    Load model from S3 if USE_EXISTING_MODEL is True, otherwise train a new one.
    Returns:
        model_wrapper: An instance of ViTModelWrapper or ResNetModelWrapper with a trained or loaded model.
    """
    if MODEL_TYPE not in MODEL_WRAPPERS:
        raise ValueError(f"Unsupported model type: {MODEL_TYPE}")
      print(f"=== dataset_path {CONFIG["dataset_path"]} ===")
    image_paths, labels = load_dataset(CONFIG["dataset_path"])
    train_dataset, val_dataset, test_dataset = split_dataset(image_paths, labels)

    model_wrapper = MODEL_WRAPPERS[MODEL_TYPE]()    
    if USE_EXISTING_MODEL:
        try:
            filename = model_wrapper.get_best_model_filename()
            s3_path = f"Models/{MODEL_TYPE}/{filename}"
            s3_manager.load_model(model_wrapper.model, s3_path)
            logger.info(f"Loaded model from s3://{s3_path}")
            return model_wrapper.model
        except Exception as e:
            logger.warning(f"Failed to load existing model. Training a new one. Error: {e}")
 

    print(f"=== Training {MODEL_TYPE} ===")
    model_wrapper = MODEL_WRAPPERS[MODEL_TYPE]() 

    trained_model = train(model_wrapper, train_dataset, val_dataset, CONFIG)
    return test_dataset, trained_model["model"]