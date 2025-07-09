from abc import ABC, abstractmethod
from datetime import datetime
from model_wrapper.models.losses.loss_factory import get_loss_fn

class BaseModelWrapper(ABC):
    def __init__(self, config):
        self.config = config
        self.model = self.get_model()
        self.init_criterion()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def preprocess(self, inputs, labels, device):
        pass

    @abstractmethod
    def forward_pass(self, inputs):
        pass
    
    def init_criterion(self):
        loss_type = self.config.get("loss_type", "bce")
        loss_params = self.config.get("loss_params", {})
        self.criterion = get_loss_fn(loss_type, loss_params)

    def get_best_model_filename(self):
        return f"{self.__class__.__name__}_best.pth"
    
    def predict(self, outputs):
        return self.criterion.predict(outputs)