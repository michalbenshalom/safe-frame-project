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

    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME)
    if 'Contents' not in response:
        return {}

    zip_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.zip')]
    extracted_files = {} if zip_files else None
    if zip_files:
        zip_files = [zip_files[1]]
    for file_name in zip_files:
        # Download file from S3
        obj = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=file_name)
        zip_content = obj['Body'].read()

        # Extract ZIP file in memory
        with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
            extracted_files[file_name] = {name: z.read(name) for name in z.namelist()}

    return extracted_files

if __name__ == "__main__":
    files = download_and_extract_s3_files()
    print(files.keys())  