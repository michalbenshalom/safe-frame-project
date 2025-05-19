from fastapi import FastAPI
from src.data_management.s3_manager import download_and_extract_s3_files
from src.data_management.validate import validate_video_files


app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello from SafeFrame API"}

@app.get("/validate-videos")
def validate():
    files = download_and_extract_s3_files()
    results = validate_video_files(files)
    return results
