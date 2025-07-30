from loguru import logger
from training_evaluation.Test.test_stage import test_model
from model_wrapper.data.image_utils import load_dataset
from model_wrapper.data.dataset_loader import split_dataset
from config import CONFIG,MODEL_TYPE,USE_EXISTING_MODEL
from utils.ModelsTypes import MODEL_WRAPPERS
from .train.trainer import train
from utils.s3_model_manager import S3ModelManager
from torch.utils.tensorboard import SummaryWriter

s3_manager = S3ModelManager()


def run_models_pipeline():
    """
    Runs the full pipeline, returning test dataset and the trained or loaded model.
    Returns:
        Tuple[test_dataset, model]: Evaluation dataset and the model instance.
    """
    result = get_or_train_model()
    
    # אם זה המשך אימון, נקבל 4 ערכים: train_dataset, val_dataset, test_dataset, model
    if CONFIG.get("continue_training", False) and len(result) == 4:
        train_dataset, val_dataset, test_dataset, model = result
        # המשך אימון המודל הקיים
        print(f"=== Continuing training {CONFIG['model_type']} ===")
        trained_model = train(train_dataset, val_dataset, CONFIG, initial_model=model)
        test_dataset = result[2]  # test_dataset
        model = trained_model["model"]
    else:
        # אימון רגיל או טעינת מודל לבדיקה
        test_dataset, model = result
    
    test_result = test_model(model, test_dataset)
    log_dir = CONFIG.get("tensorboard_log_dir", "./runs")
    writer = SummaryWriter(log_dir=log_dir)
    final_step = CONFIG.get("epochs", 5)
    writer.add_scalar("Test/Accuracy", test_result["accuracy"] * 100, final_step)
    writer.add_scalar("Test/Precision", test_result["precision"] * 100, final_step)
    writer.add_scalar("Test/Recall", test_result["recall"] * 100, final_step)
    writer.add_scalar("Test/F1_Score", test_result["f1_score"] * 100, final_step)

    writer.add_hparams(
        {
            "lr": CONFIG["learning_rate"],
            "epochs": CONFIG["epochs"],
            "model": CONFIG["model_type"]
        },
        {
            "accuracy": test_result["accuracy"],
            "f1_score": test_result["f1_score"],
            "precision": test_result["precision"],
            "recall": test_result["recall"]
        }
    )
    writer.close()
    return test_result



def get_or_train_model():
    """
    Load model from S3 if USE_EXISTING_MODEL is True or continue_training is True, otherwise train a new one.
    Returns:
        model_wrapper: An instance of ViTModelWrapper or ResNetModelWrapper with a trained or loaded model.
    """
    if CONFIG["model_type"] not in MODEL_WRAPPERS:
        raise ValueError(f"Unsupported model type: {CONFIG['model_type']}")
    
    image_paths, labels = load_dataset(CONFIG["dataset_path"])
    train_dataset, val_dataset, test_dataset = split_dataset(image_paths, labels)

    model_wrapper = MODEL_WRAPPERS[CONFIG["model_type"]]()    
    
    # בדוק אם צריך לטעון מודל קיים (לבדיקה או להמשך אימון)
    if USE_EXISTING_MODEL or CONFIG.get("continue_training", False):
        try:
            filename = model_wrapper.get_best_model_filename()
            s3_path = f"Models/{CONFIG['model_type']}/{filename}"
            s3_manager.load_model(model_wrapper.model, s3_path)
            logger.info(f"Loaded model from s3://{s3_path}")
            
            # אם זה להמשך אימון, החזר את המודל עם הדאטהסטים לאימון
            if CONFIG.get("continue_training", False):
                return train_dataset, val_dataset, test_dataset, model_wrapper.model
            else:
                # אם זה רק לבדיקה, החזר רק את test dataset
                return test_dataset, model_wrapper.model
                
        except Exception as e:
            logger.warning(f"Failed to load existing model. Training a new one. Error: {e}")
 
    print(f"=== Training {CONFIG['model_type']} ===")
    trained_model = train(train_dataset, val_dataset, CONFIG)
    return test_dataset, trained_model["model"]