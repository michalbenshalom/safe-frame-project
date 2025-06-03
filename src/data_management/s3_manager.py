import boto3
import os
from dotenv import load_dotenv

# טען משתני סביבה מהקובץ .env
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def stream_s3_videos():
    """
    Generator שמזרים קובצי וידאו מהתיקיות ב־S3 בזה אחר זה.
    מחזיר בכל איטרציה:
    - שם התיקייה (למשל: Arson)
    - שם הקובץ (video1.mp4)
    - תוכן הקובץ (bytes)
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    paginator = s3.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=AWS_BUCKET_NAME, Prefix="DCSASS Dataset/"):
        for obj in page.get('Contents', []):
            key = obj['Key']

            if key.endswith('/') or 'Labels' in key:
                continue  
            parts = key.split('/')
            if len(parts) < 3:
                continue  # צפה למבנה DCSASS Dataset/קטגוריה/קובץ

            folder = parts[1]         # למשל Arson
            file_name = parts[2]      # למשל video123.mp4

            response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
            file_bytes = response['Body'].read()

            yield folder, file_name, file_bytes
