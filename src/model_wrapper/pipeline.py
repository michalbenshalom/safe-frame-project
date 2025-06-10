# pipeline.py

from model_wrapper.models.resnet_model import trainModel as trainResNet
from model_wrapper.models.vit_model import trainModel as trainViT

def run_models_pipeline(model_type):
    """
    Run the pipeline based on the selected model type.

    Args:
        model_type (str): The type of model to train ("ViT" or "ResNet").
    """
    if model_type == "ViT":
        trainViT()
    elif model_type == "ResNet":
        trainResNet()
    else:
        print("Invalid model type. Please choose 'ViT' or 'ResNet'.")

# Example usage
if __name__ == "__main__":
    model_type = input("Enter model type (ViT/ResNet): ")
    run_models_pipeline(model_type)
