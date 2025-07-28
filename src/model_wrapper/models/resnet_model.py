from torch import nn
from transformers import ResNetForImageClassification
from src.config import CONFIG
from .base_model_wrapper import BaseModelWrapper

class ResNetModelWrapper(BaseModelWrapper):
    def get_model(self):
        hf_name = CONFIG["model_hf_names"]["resnet"]
        model = ResNetForImageClassification.from_pretrained(
            hf_name,
            num_labels=CONFIG["num_classes"],
            ignore_mismatched_sizes=True
        )
        
        classifier_module = getattr(model, 'classifier', None)
        if isinstance(classifier_module, nn.Sequential) and len(classifier_module) > 1:
            in_features = classifier_module[1].in_features
            model.classifier[1] = nn.Linear(in_features, CONFIG["num_classes"])
        elif isinstance(classifier_module, nn.Linear):
            model.classifier = nn.Linear(classifier_module.in_features, CONFIG["num_classes"])
        elif hasattr(model, 'fc'):
            model.fc = nn.Linear(model.fc.in_features, CONFIG["num_classes"])
        return model


    def preprocess(self, inputs, labels, device):
        inputs = inputs.to(device)
        labels = labels.to(device)

        if inputs.ndim == 3:
            inputs = inputs.unsqueeze(0)  

        # עבור binary classification עם BCE loss - צריך float
        labels = labels.view(-1, 1).float()
        return inputs, labels

    def forward_pass(self, inputs):
        output = self.model(inputs)
        return output.logits if hasattr(output, 'logits') else output

    
