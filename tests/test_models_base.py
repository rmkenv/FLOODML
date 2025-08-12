
"""
Test cases for base model functionality.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from floodml.models.base import BaseFloodModel


class TestBaseFloodModel:
    """Test cases for BaseFloodModel class."""
    
    def test_base_model_initialization(self):
        """Test base model initialization."""
        model = BaseFloodModel()
        
        assert model.model is None
        assert model.is_trained is False
        assert model.feature_names == []
        assert model.target_name is None
    
    def test_base_model_fit_not_implemented(self):
        """Test that fit method raises NotImplementedError."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0, 1])
        
        with pytest.raises(NotImplementedError):
            model.fit(X, y)
    
    def test_base_model_predict_not_implemented(self):
        """Test that predict method raises NotImplementedError."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature1", "feature2"])
        
        with pytest.raises(NotImplementedError):
            model.predict(X)
    
    def test_validate_input_data_valid(self):
        """Test input validation with valid data."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0, 1])
        
        # Should not raise any exception
        model._validate_input_data(X, y)
    
    def test_validate_input_data_mismatched_lengths(self):
        """Test input validation with mismatched X and y lengths."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0])  # Different length
        
        with pytest.raises(ValueError, match="X and y must have the same number of samples"):
            model._validate_input_data(X, y)
    
    def test_validate_input_data_empty(self):
        """Test input validation with empty data."""
        model = BaseFloodModel()
        X = pd.DataFrame(columns=["feature1", "feature2"])
        y = pd.Series(dtype=float)
        
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            model._validate_input_data(X, y)
    
    def test_validate_input_data_nan_values(self):
        """Test input validation with NaN values."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, np.nan], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0, 1])
        
        with pytest.raises(ValueError, match="Input data contains NaN values"):
            model._validate_input_data(X, y)
    
    def test_validate_input_data_infinite_values(self):
        """Test input validation with infinite values."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, np.inf], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0, 1])
        
        with pytest.raises(ValueError, match="Input data contains infinite values"):
            model._validate_input_data(X, y)
    
    def test_validate_prediction_input_not_trained(self):
        """Test prediction validation when model is not trained."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2]], columns=["feature1", "feature2"])
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            model._validate_prediction_input(X)
    
    def test_validate_prediction_input_feature_mismatch(self):
        """Test prediction validation with mismatched features."""
        model = BaseFloodModel()
        model.is_trained = True
        model.feature_names = ["feature1", "feature2"]
        
        # Different feature names
        X = pd.DataFrame([[1, 2]], columns=["feature1", "feature3"])
        
        with pytest.raises(ValueError, match="Feature names do not match training data"):
            model._validate_prediction_input(X)
    
    def test_get_feature_importance_not_implemented(self):
        """Test that get_feature_importance raises NotImplementedError."""
        model = BaseFloodModel()
        
        with pytest.raises(NotImplementedError):
            model.get_feature_importance()
    
    def test_save_model_not_implemented(self):
        """Test that save_model raises NotImplementedError."""
        model = BaseFloodModel()
        
        with pytest.raises(NotImplementedError):
            model.save_model("test_path.pkl")
    
    def test_load_model_not_implemented(self):
        """Test that load_model raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            BaseFloodModel.load_model("test_path.pkl")
    
    @patch('floodml.models.base.BaseFloodModel.fit')
    def test_model_training_workflow(self, mock_fit):
        """Test the general workflow of model training."""
        model = BaseFloodModel()
        X = pd.DataFrame([[1, 2], [3, 4]], columns=["feature1", "feature2"])
        y = pd.Series([0, 1])
        
        # Mock the fit method to avoid NotImplementedError
        mock_fit.return_value = model
        
        # Test that fit is called
        result = model.fit(X, y)
        mock_fit.assert_called_once_with(X, y)
        assert result == model
    
    def test_model_string_representation(self):
        """Test string representation of model."""
        model = BaseFloodModel()
        str_repr = str(model)
        
        assert "BaseFloodModel" in str_repr
        assert "is_trained=False" in str_repr
    
    def test_model_repr(self):
        """Test repr of model."""
        model = BaseFloodModel()
        repr_str = repr(model)
        
        assert "BaseFloodModel" in repr_str
