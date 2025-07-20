from fastapi import FastAPI, UploadFile, File, Query
from inference_pipeline.pipeline import predict_from_video_file
from src.data_management.data_pipeline import process_and_validate_videos
from training_evaluation.pipeline import run_models_pipeline
from src.config import RELOAD_DATASET, CONFIG


app = FastAPI()
 
@app.get("/")
def root(
    dataset_percent: float = Query(5, description="dataset percent"),
    val_size: float = Query(0.005, description="Validation set size"),
    test_size: float = Query(0.99, description="Test set size"),
    epochs: int = Query(2, description="Number of epochs"),
    model_type: str = Query("RESNET", description="Model type"),
    loss_type: str = Query("bce", description="Loss type"),
):
    try:
        CONFIG["model_type"] = model_type
        CONFIG["val_size"] = val_size
        CONFIG["test_size"] = test_size
        CONFIG["epochs"] = epochs
        CONFIG["dataset_percent"] = dataset_percent
        CONFIG["loss_type"] = loss_type

        if RELOAD_DATASET:
            process_and_validate_videos()
        result = run_models_pipeline()
        if result is None:
            return {"error": "run_models_pipeline returned None. Check your model training pipeline."}
        return {
            "message": f"Training completed using {CONFIG["model_type"]}",
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
