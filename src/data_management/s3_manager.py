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
    Connect to S3, download 3 ZIP files, and extract them into memory.
    """
    # Connect to S3
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    # List of files to download
    files_to_download = ["DCSASSDataSet.zip", "Guns-Knives Object Detection.zip"]
    extracted_files = {}

    for file_name in files_to_download:
        # Download file from S3
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=file_name)
        zip_content = response["Body"].read()

        # Extract ZIP file in memory
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            extracted_files[file_name] = {name: z.read(name) for name in z.namelist()}

    return extracted_files


if __name__ == "__main__":
    files = download_and_extract_s3_files()
    print(files.keys())  # הדפס את שמות הקבצים שהורדו