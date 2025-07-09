
from config import CONFIG
from model_wrapper.models.resnet_model import ResNetModelWrapper
from model_wrapper.models.vit_model import ViTModelWrapper



MODEL_WRAPPERS = {
    "VIT": lambda: ViTModelWrapper(CONFIG),
    "RESNET": lambda: ResNetModelWrapper(CONFIG),
}