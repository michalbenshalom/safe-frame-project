from torch import nn
from transformers import ResNetForImageClassification
from config import CONFIG
from .base_model_wrapper import BaseModelWrapper

class ResNetModelWrapper(BaseModelWrapper):
    def get_model(self):
        hf_name = CONFIG["model_hf_names"]["resnet"]
        model = ResNetForImageClassification.from_pretrained(
            hf_name,
            num_labels=CONFIG["num_classes"],
            ignore_mismatched_sizes=True
        )
        try:
            in_features = model.classifier[1].in_features  # אם זו Sequential
            model.classifier[1] = nn.Linear(in_features, CONFIG["num_classes"])
        except (AttributeError, IndexError, TypeError):
            try:
                in_features = model.classifier.in_features  # אם זו Linear ישירה
                model.classifier = nn.Linear(in_features, CONFIG["num_classes"])
            except AttributeError:
                model.fc = nn.Linear(model.fc.in_features, CONFIG["num_classes"])  # לגיבוי
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

    
