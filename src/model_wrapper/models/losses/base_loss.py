import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        raise NotImplementedError("Loss must implement forward method.")
    
    def predict(self, outputs):
        raise NotImplementedError("Loss must implement predict method.")