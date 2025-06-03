import boto3
import zipfile
import io
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")

def download_and_extract_s3_files():
    """
    Connect to S3, download all folders and their video files, and organize them in memory.
    """
    # Connect to S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    # List all objects in the bucket
    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME)
    if 'Contents' not in response:
        return {}

    # Organize files by folder
    folder_structure = {}
    for obj in response['Contents']:
        key = obj['Key']
        if key.endswith('/'):  # Skip folder markers
            continue

        # Extract folder name and file name
        parts = key.split('/')
        if len(parts) > 1:
            folder_name = parts[0]
            file_name = '/'.join(parts[1:])
        else:
            folder_name = "root"
            file_name = parts[0]

        # Download the file
        obj = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
        file_content = obj['Body'].read()

        # Organize files in a dictionary
        if folder_name not in folder_structure:
            folder_structure[folder_name] = {}
        folder_structure[folder_name][file_name] = file_content

    return folder_structure

if __name__ == "__main__":
    files = download_and_extract_s3_files()
    for folder, videos in files.items():
        print(f"Folder: {folder}, Files: {list(videos.keys())}")