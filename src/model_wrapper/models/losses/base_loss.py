import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        raise NotImplementedError("Loss must implement forward method.")