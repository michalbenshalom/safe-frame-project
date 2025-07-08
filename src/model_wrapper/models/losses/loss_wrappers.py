from regex import F
import torch
import torch.nn as nn
from .base_loss import BaseLoss

class BCEWithLogitsLossWrapper(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(**kwargs)

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)

    def predict(self, outputs):
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).float()
    
class CrossEntropyLossWrapper(BaseLoss):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(**kwargs)

    def forward(self, outputs, targets):
        return self.loss_fn(outputs, targets)
    
    def predict(self, outputs):
        return torch.argmax(outputs, dim=1)
    
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        
    def predict(self, outputs):
        probs = torch.sigmoid(outputs)
        return (probs > 0.5).float()