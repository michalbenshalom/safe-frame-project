from fastapi import FastAPI, UploadFile, File, Query
from inference_pipeline.pipeline import predict_from_video_file
from src.data_management.data_pipeline import process_and_validate_videos
from training_evaluation.pipeline import run_models_pipeline
from src.config import RELOAD_DATASET, MODEL_TYPE, CONFIG


app = FastAPI()
 
@app.get("/")
def root(
    dataset_percent: float = Query(5, description="dataset percent"),
    val_size: float = Query(0.005, description="Validation set size"),
    test_size: float = Query(0.99, description="Test set size"),
    epochs: int = Query(2, description="Number of epochs")
):
    try:
        CONFIG["val_size"] = val_size
        CONFIG["test_size"] = test_size
        CONFIG["epochs"] = epochs
        CONFIG["dataset_percent"] = dataset_percent
        
        if RELOAD_DATASET:
            process_and_validate_videos()
        result = run_models_pipeline()
        if result is None:
            return {"error": "run_models_pipeline returned None. Check your model training pipeline."}
        return {
            "message": f"Training completed using {MODEL_TYPE}",
            "model": MODEL_TYPE,
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
