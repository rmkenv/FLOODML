"""
Machine learning models for FloodML
"""

from .base import BaseFloodModel
from .random_forest import RandomForestFloodModel

try:
    from .lstm import LSTMFloodModel
except ImportError:
    LSTMFloodModel = None

try:
    from .ensemble import EnsembleFloodModel
except ImportError:
    EnsembleFloodModel = None

try:
    from .evaluation import ModelEvaluator
except ImportError:
    ModelEvaluator = None

__all__ = [
    "BaseFloodModel",
    "RandomForestFloodModel"
]

if LSTMFloodModel is not None:
    __all__.append("LSTMFloodModel")
if EnsembleFloodModel is not None:
    __all__.append("EnsembleFloodModel")
if ModelEvaluator is not None:
    __all__.append("ModelEvaluator")
