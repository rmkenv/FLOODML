"""
FloodML - Machine Learning for Flood Prediction

A Python package for predicting flood events using USGS streamflow and NWS weather data.
"""

__version__ = "0.1.0"
__author__ = "FloodML Contributors"
__email__ = "info@floodml.org"

from .prediction.forecaster import FloodPredictor
from .data.usgs import USGSCollector
from .data.nws import NWSCollector
from .models.random_forest import RandomForestFloodModel
from .models.lstm import LSTMFloodModel
from .models.ensemble import EnsembleFloodModel

# Main API exports
__all__ = [
    "FloodPredictor",
    "USGSCollector", 
    "NWSCollector",
    "RandomForestFloodModel",
    "LSTMFloodModel", 
    "EnsembleFloodModel",
]