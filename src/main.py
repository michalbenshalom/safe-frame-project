from fastapi import FastAPI
from data_management.data_pipeline import process_and_validate_videos
from config import CONFIG
import pdb

app = FastAPI()

@app.get("/")
def root():

    if CONFIG["reload_dataset"]:
       process_and_validate_videos()

    result = run_models_pipeline(CONFIG["model_type"])
    
    #model_result = 
    return {"message": "Hello from SafeFrame API", "process_result": result}
