import os
import json
import io
import boto3
import torch
from datetime import datetime
from loguru import logger


class S3ModelManager:
    def __init__(self):
        self.aws_access_key = os.getenv("AWS_ACCESS_KEY")
        self.aws_secret_key = os.getenv("AWS_SECRET_KEY")
        self.bucket_name = os.getenv("AWS_BUCKET_NAME")
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key,
            aws_secret_access_key=self.aws_secret_key
        )

    def save_model(self, model_state_dict, s3_path: str):
        """
        שומר state_dict של מודל ל־S3.
        """
        buffer = io.BytesIO()
        torch.save(model_state_dict, buffer)
        buffer.seek(0)

        self.s3.upload_fileobj(buffer, self.bucket_name, s3_path)
        logger.info(f"[S3] Model saved to s3://{self.bucket_name}/{s3_path}")

    def load_model(self, model, s3_path: str):
        """
        טוען state_dict מתוך S3 למודל קיים.
        """
        try:
            buffer = io.BytesIO()
            self.s3.download_fileobj(self.bucket_name, s3_path, buffer)
            buffer.seek(0)
            state_dict = torch.load(buffer, map_location=torch.device("cpu"))
            model.load_state_dict(state_dict)
            logger.info(f"[S3]Model loaded from s3://{self.bucket_name}/{s3_path}")
            return model
        except Exception as e:
            logger.error(f"[S3] Failed to load model: {e}")
            raise RuntimeError(f"Unable to load model from S3: {e}")

    def save_history(self, history: list, checkpoint_dir: str = "./checkpoints/"):
        """
        שומר היסטוריית אימון כ־JSON מקומי וגם מעלה ל־S3.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"train_history_{timestamp}.json"
        local_path = os.path.join(checkpoint_dir, filename)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # שמירה מקומית
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Saved history locally: {local_path}")

        # העלאה ל־S3
        try:
            buffer = io.BytesIO(json.dumps(history, indent=4).encode("utf-8"))
            s3_key = f"History/{filename}"
            self.s3.upload_fileobj(buffer, self.bucket_name, s3_key)
            logger.info(f"[S3] Uploaded history to s3://{self.bucket_name}/{s3_key}")
        except Exception as e:
            logger.warning(f"[S3] Failed to upload history: {e}")
