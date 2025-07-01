from transformers import ResNetForImageClassification

from src.config import CONFIG

def get_resnet_model():
    hf_name = CONFIG["model_hf_names"]["vit"]
    model = ResNetForImageClassification.from_pretrained(hf_name, num_labels=CONFIG["num_classes"],ignore_mismatched_sizes=True)
    return model

