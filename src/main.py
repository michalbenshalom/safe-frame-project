from fastapi import FastAPI, UploadFile, File, Query
from inference_pipeline.pipeline import predict_from_video_file
from src.data_management.data_pipeline import process_and_validate_videos
from training_evaluation.pipeline import run_models_pipeline
from src.config import RELOAD_DATASET, CONFIG


app = FastAPI()
 
@app.get("/")
def root(
    dataset_percent: float = Query(None, description="dataset percent"),
    val_size: float = Query(None, description="Validation set size"),
    test_size: float = Query(None, description="Test set size"),
    epochs: int = Query(None, description="Number of epochs"),
    model_type: str = Query(None, description="Model type"),
    loss_type: str = Query(None, description="Loss type"),
    continue_training: bool = Query(None, description="Continue training existing model from S3"),
    train_strategy: str = Query(None, description="Training strategy: 'full' or 'top_layers'"),
):
    try:
        # עדכון CONFIG רק אם הפרמטר נשלח (לא None)
        if model_type is not None:
            CONFIG["model_type"] = model_type
        if val_size is not None:
            CONFIG["val_size"] = val_size
        if test_size is not None:
            CONFIG["test_size"] = test_size
        if epochs is not None:
            CONFIG["epochs"] = epochs
        if dataset_percent is not None:
            CONFIG["dataset_percent"] = dataset_percent
        if loss_type is not None:
            CONFIG["loss_type"] = loss_type
        if continue_training is not None:
            CONFIG["continue_training"] = continue_training
        if train_strategy is not None:
            CONFIG["train_strategy"] = train_strategy

        if RELOAD_DATASET:
            process_and_validate_videos()
        result = run_models_pipeline()
        if result is None:
            return {"error": "run_models_pipeline returned None. Check your model training pipeline."}
        return {
            "message": f"Training completed using {CONFIG['model_type']}",
            "model": CONFIG["model_type"],
            "metrics": {
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1_score": result.get("f1_score")
            }
        }
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/predict_video")
async def predict_video(file: UploadFile = File(...)):
    try:
        result = await predict_from_video_file(file)
        return result
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
