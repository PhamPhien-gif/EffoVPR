__all__ = ["EffoVPR", "CosFace", "build_effovpr_model"]

from .models.effovpr import EffoVPR
from .models.cosface import CosFace


def build_effovpr_model(*args, **kwargs):
    return EffoVPR(*args, **kwargs)
