from transformers import ViTForImageClassification

def trainModel(num_classes, hf_name):
    model = ViTForImageClassification.from_pretrained(hf_name, num_labels=num_classes)
    return model
  