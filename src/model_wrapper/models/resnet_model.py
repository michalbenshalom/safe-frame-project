from transformers import ResNetForImageClassification

def get_resnet_model(num_classes, hf_name):
    model = ResNetForImageClassification.from_pretrained(hf_name, num_labels=num_classes)
    return model