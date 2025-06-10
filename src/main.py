from fastapi import FastAPI
from data_management.data_pipeline import process_and_validate_videos
import pdb
from config import RELOAD_DATASET

app = FastAPI()

@app.get("/")
def root():

    if RELOAD_DATASET:
       process_and_validate_videos()


    
    #model_result = 
    return {"message": "Hello from SafeFrame API", "process_result": result}
