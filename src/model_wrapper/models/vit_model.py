from transformers import ViTForImageClassification
from src.config import CONFIG

def trainModel():
    hf_name = CONFIG["model_hf_names"]["vit"]
    model = ViTForImageClassification.from_pretrained(hf_name, num_labels=CONFIG["num_classes"])
    return model
