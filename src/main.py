from fastapi import FastAPI
from data_management.data_pipeline import process_and_validate_videos
import pdb


app = FastAPI()

@app.get("/")
def root():
    result = process_and_validate_videos()
    return {"message": "Hello from SafeFrame API", "process_result": result}
