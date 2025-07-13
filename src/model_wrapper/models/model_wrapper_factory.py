from model_wrapper.models.vit_model import ViTModelWrapper
from model_wrapper.models.resnet_model import ResNetModelWrapper

class ModelWrapperFactory:
    @staticmethod
    def create(model_type: str, config: dict):
        model_type = model_type.upper()
        if model_type == "VIT":
            return ViTModelWrapper(config)
        elif model_type == "RESNET":
            return ResNetModelWrapper(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
