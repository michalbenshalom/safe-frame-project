import os
import json
import boto3
import io
import torch
from datetime import datetime
from loguru import logger


def save_best_model(model_state_dict, bucket_name, s3_path, aws_access_key, aws_secret_key):
    """
    שומר את המודל ל־S3 ישירות מה־state_dict.
    """
    buffer = io.BytesIO()
    torch.save(model_state_dict, buffer)
    buffer.seek(0)

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    s3.upload_fileobj(buffer, bucket_name, s3_path)
    logger.info(f"[S3] Saved best model to s3://{bucket_name}/{s3_path}")


def save_train_history(history, config):
    """
    שומר את היסטוריית האימון כ־JSON מקומית + ב־S3.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"train_history_{timestamp}.json"
    local_dir = config.get("checkpoint_dir", "./checkpoints/")
    local_path = os.path.join(local_dir, filename)

    # שמירה מקומית
    os.makedirs(local_dir, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)
    logger.info(f"Saved training history locally: {local_path}")

    # שמירה ל-S3 (אם מוגדר)
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
        )
        buffer = io.BytesIO(json.dumps(history, indent=4).encode("utf-8"))
        s3_key = f"History/{filename}"
        s3.upload_fileobj(buffer, os.getenv("AWS_BUCKET_NAME"), s3_key)
        logger.info(f"Uploaded training history to s3://{os.getenv('AWS_BUCKET_NAME')}/{s3_key}")
    except Exception as e:
        logger.warning(f"Failed to upload history to S3: {e}")
