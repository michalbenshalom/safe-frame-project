import torch
from abc import ABC, abstractmethod

class BaseModelWrapper(ABC):
    def __init__(self, config):
        self.config = config
        self.model = self.get_model()

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def preprocess(self, inputs, labels, device):
        pass

    @abstractmethod
    def forward_pass(self, inputs):
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    @abstractmethod
    def generate_model_filename(self):
        pass