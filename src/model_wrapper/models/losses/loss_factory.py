from .loss_wrappers import (
    BCEWithLogitsLossWrapper,
    CrossEntropyLossWrapper,
    FocalLoss
)

from .loss_wrappers import FocalLoss

def get_loss_fn(loss_type, params={}):
    if loss_type == "bce":
        return BCEWithLogitsLossWrapper(**params)
    elif loss_type == "cross_entropy":
        return CrossEntropyLossWrapper(**params)
    elif loss_type == "focal":
        gamma = params.get("gamma", 2.0)
        alpha = params.get("alpha", 0.25)
        return FocalLoss(gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

