import boto3
import zipfile
import io
from dotenv import load_dotenv
import os
import tempfile


# Load environment variables from .env file
load_dotenv()

# Get AWS credentials from environment variables
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")




def download_and_extract_s3_files_streamed():
    print("📡 Connecting to S3...")
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    response = s3.list_objects_v2(Bucket=AWS_BUCKET_NAME)
    if 'Contents' not in response:
        print("⚠️ No files found in bucket.")
        return {}

    zip_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].lower().endswith('.zip')]
    extracted_files = {}

    for i, file_key in enumerate(zip_files, start=1):
        print(f"\n📥 ({i}/{len(zip_files)}) Downloading: {file_key}")

        # הורדת הקובץ ל־disk זמני
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            s3.download_fileobj(Bucket=AWS_BUCKET_NAME, Key=file_key, Fileobj=tmp)
            tmp_path = tmp.name

        # פתיחה כ־zip רגיל
        print(f"📦 Extracting {file_key}")
        with zipfile.ZipFile(tmp_path, 'r') as z:
            extracted_files[file_key] = {
                name: z.read(name) for name in z.namelist()
            }

        os.remove(tmp_path)

    return extracted_files




if __name__ == "__main__":
    files = download_and_extract_s3_files()
    print(files.keys())  # הדפס את שמות הקבצים שהורדו