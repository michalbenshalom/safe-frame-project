from fastapi import FastAPI
from data_management.data_pipeline import process_and_validate_videos
from training_evaluation.pipeline import run_models_pipeline
from config import RELOAD_DATASET, MODEL_TYPE
#from src.model_wrapper.train.test_only import test_model


app = FastAPI()
 
@app.get("/")
def root():
    try:
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

@app.get("/metrics")
def metrics():
    # weights = "resnet18_trained.pth" if MODEL_TYPE.lower() == "resnet" else "vit_b16_trained.pth"
    # try:
    #     result = test_model(model_type=MODEL_TYPE, weights_path=weights)
    #     return {
    #         "model": MODEL_TYPE,
    #         "metrics": result
    #     }
    # except Exception as e:
    #     return {"error": str(e)}
    return {"message": "Metrics endpoint is not implemented yet."}

