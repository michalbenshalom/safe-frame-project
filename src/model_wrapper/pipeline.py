# pipeline.py

from src.model_wrapper.models.resnet_model import trainModel as trainResNet
from src.model_wrapper.models.vit_model import trainModel as trainViT
import sys

def run_models_pipeline(model_type):
    """
    Run the pipeline based on the selected model type.

    Args:
        model_type (str): The type of model to train ("ViT" or "ResNet").

    Returns:
        dict: Training/evaluation metrics or None if invalid type.
    """
    if model_type.lower() == "vit":
        return trainViT()
    elif model_type.lower() == "resnet":
        return trainResNet()
    else:
        print("Invalid model type. Please choose 'ViT' or 'ResNet'.")
        return None

# הרצה: python pipeline.py vit

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1].strip().lower()
    else:
        model_type = input("Enter model type (ViT/ResNet): ").strip().lower()

    run_models_pipeline(model_type)



