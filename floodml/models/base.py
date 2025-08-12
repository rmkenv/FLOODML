"""
Base class for flood prediction models
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import joblib
import structlog

logger = structlog.get_logger()


class BaseFloodModel(ABC):
    """
    Abstract base class for flood prediction models
    
    Defines the interface that all flood prediction models must implement.
    Provides common functionality for model management, evaluation, and persistence.
    """
    
    def __init__(self, name: str):
        """
        Initialize base flood model
        
        Parameters
        ----------
        name : str
            Name of the model
        """
        self.name = name
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self.training_history = {}
        self.metadata = {
            'model_type': self.__class__.__name__,
            'created_at': pd.Timestamp.now(),
            'version': '0.1.0'
        }
        
        logger.info(f"Initialized {name} model")
    
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'BaseFloodModel':
        """
        Fit the model to training data
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        **kwargs
            Additional model-specific parameters
            
        Returns
        -------
        BaseFloodModel
            Fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Predictions
        """
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities (for classification models)
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
            
        Returns
        -------
        np.ndarray
            Prediction probabilities
        """
        # Default implementation for regression models
        predictions = self.predict(X)
        
        # Convert regression predictions to probabilities
        # This is a simple sigmoid transformation - override in subclasses for better methods
        probs = 1 / (1 + np.exp(-predictions))
        return np.column_stack([1 - probs, probs])
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Get feature importance scores
        
        Returns
        -------
        pd.Series or None
            Feature importance scores if available
        """
        if not self.is_fitted:
            logger.warning("Model must be fitted before getting feature importance")
            return None
        
        # Default implementation - override in subclasses
        if hasattr(self.model, 'feature_importances_'):
            return pd.Series(
                self.model.feature_importances_,
                index=self.feature_names,
                name='importance'
            ).sort_values(ascending=False)
        
        return None
    
    def save(self, filepath: str):
        """
        Save model to disk
        
        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted:
            logger.warning("Saving unfitted model")
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'metadata': self.metadata
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'BaseFloodModel':
        """
        Load model from disk
        
        Parameters
        ----------
        filepath : str
            Path to the saved model
            
        Returns
        -------
        BaseFloodModel
            Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(model_data['name'])
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_names = model_data['feature_names']
        instance.training_history = model_data['training_history']
        instance.metadata = model_data['metadata']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns
        -------
        dict
            Model information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'num_features': len(self.feature_names),
            'feature_names': self.feature_names.copy(),
            'metadata': self.metadata.copy(),
            'training_history': self.training_history.copy()
        }
    
    def _validate_input(self, X: pd.DataFrame, stage: str = "prediction"):
        """
        Validate input data
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
        stage : str
            Stage of processing ("training" or "prediction")
        """
        if X.empty:
            raise ValueError("Input data is empty")
        
        if stage == "prediction" and self.is_fitted:
            # Check feature consistency
            if set(X.columns) != set(self.feature_names):
                missing_features = set(self.feature_names) - set(X.columns)
                extra_features = set(X.columns) - set(self.feature_names)
                
                error_msg = []
                if missing_features:
                    error_msg.append(f"Missing features: {missing_features}")
                if extra_features:
                    error_msg.append(f"Extra features: {extra_features}")
                
                raise ValueError("; ".join(error_msg))
            
            # Reorder columns to match training order
            X = X[self.feature_names]
        
        # Check for infinite values
        if np.any(np.isinf(X.values)):
            logger.warning("Input contains infinite values")
        
        # Check for excessive missing values
        missing_pct = X.isnull().sum() / len(X) * 100
        high_missing = missing_pct[missing_pct > 50]
        if not high_missing.empty:
            logger.warning(f"Features with >50% missing values: {high_missing.to_dict()}")
    
    def _prepare_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for model input
        
        Parameters
        ----------
        X : pd.DataFrame
            Input features
            
        Returns
        -------
        np.ndarray
            Prepared feature array
        """
        # Basic preparation - override in subclasses for model-specific preprocessing
        return X.fillna(0).values
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training process
        
        Returns
        -------
        dict
            Training summary
        """
        if not self.training_history:
            return {"message": "No training history available"}
        
        return self.training_history.copy()


class FloodPredictionResult:
    """
    Container for flood prediction results
    
    Provides structured access to prediction results with metadata.
    """
    
    def __init__(
        self,
        probability: float,
        confidence_low: Optional[float] = None,
        confidence_high: Optional[float] = None,
        forecast_time: Optional[pd.Timestamp] = None,
        model_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize prediction result
        
        Parameters
        ----------
        probability : float
            Flood probability (0-1)
        confidence_low : float, optional
            Lower confidence bound
        confidence_high : float, optional
            Upper confidence bound
        forecast_time : pd.Timestamp, optional
            Time of forecast
        model_name : str, optional
            Name of model used
        metadata : dict, optional
            Additional metadata
        """
        self.probability = probability
        self.confidence_low = confidence_low
        self.confidence_high = confidence_high
        self.forecast_time = forecast_time or pd.Timestamp.now()
        self.model_name = model_name
        self.metadata = metadata or {}
    
    def __repr__(self) -> str:
        return f"FloodPredictionResult(probability={self.probability:.3f})"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'probability': self.probability,
            'confidence_low': self.confidence_low,
            'confidence_high': self.confidence_high,
            'forecast_time': self.forecast_time,
            'model_name': self.model_name,
            'metadata': self.metadata
        }