from torch import nn
from transformers import ViTForImageClassification
from config import CONFIG
from .base_model_wrapper import BaseModelWrapper

class ViTModelWrapper(BaseModelWrapper):
    def get_model(self):
        hf_name = CONFIG["model_hf_names"]["vit"]
        model = ViTForImageClassification.from_pretrained(hf_name, num_labels=CONFIG["num_classes"])
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.config.num_labels = 1
        return model

    def preprocess(self, inputs, labels, device):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)  

        labels = labels.view(-1, 1).float()
        return inputs, labels

    def forward_pass(self, inputs):
        output = self.model(inputs)
        return output.logits if hasattr(output, 'logits') else output

   
