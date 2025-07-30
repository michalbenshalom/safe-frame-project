from abc import ABC, abstractmethod
from datetime import datetime
from model_wrapper.models.losses.loss_factory import get_loss_fn

class BaseModelWrapper(ABC):
    def __init__(self, config):
        self.config = config
        self.model = self.get_model()
        self.init_criterion()
        self.setup_training_strategy()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def preprocess(self, inputs, labels, device):
        pass

    @abstractmethod
    def forward_pass(self, inputs):
        pass
    
    def set_model(self, model):
        if model is None:
            raise ValueError("set_model: Received model is None.")

        from torch.nn import Module
        if not isinstance(model, Module):
            raise TypeError(f"set_model: Expected torch.nn.Module, got {type(model)}")
        self.model = model

    def init_criterion(self):
        loss_type = self.config.get("loss_type", "bce")
        loss_params = self.config.get("loss_params", {})
        self.criterion = get_loss_fn(loss_type, loss_params)

    def setup_training_strategy(self):
        """
        מגדיר אסטרטגיית אימון - אימון מלא או רק שכבות עליונות
        """
        strategy = self.config.get("train_strategy", "full")
        
        if strategy == "top_layers":
            self.freeze_backbone_layers()
            print(f"🔒 Frozen backbone layers - training only top layers")
        else:
            print(f"🔄 Training full model")

    def freeze_backbone_layers(self):
        """
        מקפיא את השכבות הבסיסיות של המודל (backbone)
        """
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            # אפשר גם להחריג שכבות ספציפיות
        if hasattr(self.model, 'classifier'):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
                

 
    def get_best_model_filename(self):
        return f"{self.__class__.__name__}_best.pth"
    
    def predict(self, outputs):
        return self.criterion.predict(outputs)