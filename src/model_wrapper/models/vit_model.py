from datetime import datetime
from torch import nn
from transformers import ViTForImageClassification
from src.config import CONFIG
from .base_model_wrapper import BaseModelWrapper

class ViTModelWrapper(BaseModelWrapper):
    def get_model(self):
        hf_name = CONFIG["model_hf_names"]["vit"]
        model = ViTForImageClassification.from_pretrained(hf_name, num_labels=CONFIG["num_classes"])
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.config.num_labels = 1
        return model

    def preprocess(self, inputs, labels, device):
        inputs, labels = inputs.to(device), labels.to(device)
        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)
        if labels.ndim == 0:
            labels = labels.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        labels = labels.float()
        return inputs, labels

    def forward_pass(self, inputs):
        output = self.model(inputs)
        return output.logits if hasattr(output, 'logits') else output

    def get_criterion(self):
        return nn.BCEWithLogitsLoss()

    def generate_model_filename(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"ViT_{timestamp}_best.pt"
