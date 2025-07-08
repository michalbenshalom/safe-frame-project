# src/model_wrapper/losses/loss_factory.py

from .loss_wrappers import (
    BCEWithLogitsLossWrapper,
    CrossEntropyLossWrapper,
    MSELossWrapper
)

from .loss_wrappers import FocalLoss

def get_loss_fn(loss_type, params={}):
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss(**params)
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(**params)
    elif loss_type == "focal":
        gamma = params.get("gamma", 2.0)
        alpha = params.get("alpha", 0.25)
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

