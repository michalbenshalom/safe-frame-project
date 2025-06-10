import boto3
import os
from dotenv import load_dotenv
from io import StringIO
import pandas as pd
import traceback

# טען משתני סביבה מהקובץ .env
load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")


def stream_s3_videos():
    """
    Generator שמזרים קובצי וידאו מתוך תיקיות האירועים (Event Folders) ב-S3.
    מחזיר בכל איטרציה:
    - קטגוריה (למשל: Abuse)
    - שם האירוע (למשל: Abuse003_x264.mp4)
    - שם הקובץ (למשל: Abuse003_x264_0.mp4)
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
            if len(parts) < 4:
                continue  # צפה למבנה DCSASS Dataset/<category>/<event_folder>/<file>

            category = parts[1]
            event_folder = parts[2]
            file_name = parts[3]

            response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
            file_bytes = response['Body'].read()

            yield category, event_folder, file_name, file_bytes



def load_labels_for_category_s3(category: str):
    
    """
    Loads a label file for a specific category from an S3 bucket and returns a dictionary mapping filenames to labels.
    Args:
        category (str): The category name whose label file should be loaded from S3.
    Returns:
        dict: A dictionary where each key is a filename (str) and each value is the corresponding label (int).
              Returns an empty dictionary if the file cannot be loaded or an error occurs.
    Notes:
        - The function connects to AWS S3 using credentials defined elsewhere.
        - It expects the label file to be in CSV format at the path 'DCSASS Dataset/Labels/{category}.csv' in the S3 bucket.
        - The CSV file should have columns: filename, category, label (no header row).
        - Empty rows are removed before processing.
        - If an error occurs (e.g., file not found, connection error), the function prints the traceback and returns an empty dictionary.
    """
    s3 = boto3.client(
        "s3",
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    key = f"DCSASS Dataset/Labels/{category}.csv"

    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=key)
        csv_content = response['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content), header=None, names=["filename", "category", "label"])

        # נקה שורות ריקות
        df = df.dropna()

        return {
            (str(row["filename"])): int(row["label"])
            for _, row in df.iterrows()
        }

    except Exception as e:
        traceback.print_exc()
        return {}