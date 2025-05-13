from fastapi import FastAPI
from src.data_management.s3_manager import download_and_extract_s3_files

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}


@app.get("/s3-files")
def get_s3_files():
    files = download_and_extract_s3_files()
    return {"files": list(files.keys())}