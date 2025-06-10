from fastapi import FastAPI
from data_management.data_pipeline import process_and_validate_videos
from model_wrapper.pipeline import run_models_pipeline, MODEL_TYPE
import pdb
from config import RELOAD_DATASET

app = FastAPI()

@app.get("/")
def root():

    if RELOAD_DATASET:
       process_and_validate_videos()

    result = run_models_pipeline(MODEL_TYPE)
    
    #model_result = 
    return {"message": "Hello from SafeFrame API", "process_result": result}
