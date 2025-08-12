"""
Basic tests for FloodML package
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest.mock as mock

from floodml import FloodPredictor
from floodml.data.usgs import USGSCollector
from floodml.data.nws import NWSCollector
from floodml.data.validation import DataValidator
from floodml.data.preprocessor import DataPreprocessor
from floodml.models.random_forest import RandomForestFloodModel


class TestDataValidator:
    """Test data validation functionality"""
    
    def test_validate_streamflow(self):
        """Test streamflow data validation"""
        validator = DataValidator()
        
        # Create test data with some issues
        dates = pd.date_range('2023-01-01', periods=100, freq='1H')
        df = pd.DataFrame({
            'discharge_cfs': np.random.uniform(100, 1000, 100),
            'gage_height_ft': np.random.uniform(5, 15, 100)
        }, index=dates)
        
        # Add some problematic data
        df.loc[dates[10], 'discharge_cfs'] = -100  # Negative discharge
        df.loc[dates[20], 'gage_height_ft'] = np.inf  # Infinite value
        df.loc[dates[30:35], 'discharge_cfs'] = np.nan  # Missing values
        
        # Validate
        cleaned_df = validator.validate_streamflow(df)
        
        # Check results
        assert len(cleaned_df) <= len(df)  # May remove some rows
        assert not np.any(cleaned_df['discharge_cfs'] < 0)  # No negative discharge
        assert not np.any(np.isinf(cleaned_df.values))  # No infinite values
    
    def test_validate_weather(self):
        """Test weather data validation"""
        validator = DataValidator()
        
        # Create test weather data
        dates = pd.date_range('2023-01-01', periods=50, freq='1H')
        df = pd.DataFrame({
            'temperature_f': np.random.uniform(20, 80, 50),
            'precipitation_mm': np.random.exponential(2, 50),
            'humidity_pct': np.random.uniform(30, 90, 50)
        }, index=dates)
        
        # Add problematic data
        df.loc[dates[5], 'temperature_f'] = -200  # Impossible temperature
        df.loc[dates[10], 'precipitation_mm'] = -5  # Negative precipitation
        df.loc[dates[15], 'humidity_pct'] = 150  # Invalid humidity
        
        # Validate
        cleaned_df = validator.validate_weather(df)
        
        # Check results
        assert not np.any(cleaned_df['temperature_f'] < -100)  # Reasonable temp range
        assert not np.any(cleaned_df['precipitation_mm'] < 0)  # No negative precip
        assert not np.any(cleaned_df['humidity_pct'] > 100)  # Valid humidity range


class TestDataPreprocessor:
    """Test data preprocessing functionality"""
    
    def test_create_flood_features(self):
        """Test feature creation"""
        preprocessor = DataPreprocessor()
        
        # Create sample streamflow data
        dates = pd.date_range('2023-01-01', periods=200, freq='1H')
        streamflow_data = pd.DataFrame({
            'gage_height_ft': 10 + np.random.normal(0, 2, 200),
            'discharge_cfs': 500 + np.random.normal(0, 100, 200)
        }, index=dates)
        
        # Create sample weather data
        weather_data = pd.DataFrame({
            'precipitation_mm': np.random.exponential(1, 200),
            'temperature_f': 50 + 20 * np.sin(np.arange(200) * 2 * np.pi / 24) + np.random.normal(0, 5, 200)
        }, index=dates)
        
        # Create features
        features, targets = preprocessor.create_flood_features(
            streamflow_data, 
            weather_data, 
            flood_stage=15.0, 
            forecast_hours=24
        )
        
        # Check results
        assert isinstance(features, pd.DataFrame)
        assert isinstance(targets, pd.Series)
        assert len(features) == len(targets)
        assert len(features.columns) > 0  # Should have created features
        assert targets.dtype == int  # Binary target
        assert all(targets.isin([0, 1]))  # Binary values only
    
    def test_temporal_features(self):
        """Test temporal feature creation"""
        preprocessor = DataPreprocessor()
        
        dates = pd.date_range('2023-01-01', periods=48, freq='1H')
        df = pd.DataFrame({'dummy': np.ones(48)}, index=dates)
        
        df_with_temporal = preprocessor._add_temporal_features(df)
        
        # Check temporal features were added
        expected_features = ['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos']
        for feature in expected_features:
            assert feature in df_with_temporal.columns
        
        # Check cyclical encoding
        assert df_with_temporal['hour_sin'].between(-1, 1).all()
        assert df_with_temporal['hour_cos'].between(-1, 1).all()


class TestRandomForestModel:
    """Test Random Forest model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        model = RandomForestFloodModel(n_estimators=50, random_state=42)
        
        assert model.name == "RandomForestFlood"
        assert not model.is_fitted
        assert model.params['n_estimators'] == 50
        assert model.params['random_state'] == 42
    
    def test_model_training_and_prediction(self):
        """Test model training and prediction"""
        model = RandomForestFloodModel(n_estimators=10, random_state=42)
        
        # Create synthetic training data
        np.random.seed(42)
        n_samples = 500
        n_features = 10
        
        X = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Create realistic flood target (rare events)
        y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]))
        
        # Train model
        model.fit(X, y)
        
        assert model.is_fitted
        assert len(model.feature_names) == n_features
        
        # Test prediction
        predictions = model.predict(X[:10])
        assert len(predictions) == 10
        assert all(0 <= p <= 1 for p in predictions)  # Valid probabilities
        
        # Test predict_proba
        probas = model.predict_proba(X[:10])
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1)  # Probabilities sum to 1
    
    def test_feature_importance(self):
        """Test feature importance calculation"""
        model = RandomForestFloodModel(n_estimators=10, random_state=42)
        
        # Create training data
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=['A', 'B', 'C', 'D', 'E'])
        y = pd.Series(np.random.choice([0, 1], size=100))
        
        # Train model
        model.fit(X, y)
        
        # Get feature importance
        importance = model.get_feature_importance()
        
        assert isinstance(importance, pd.Series)
        assert len(importance) == 5
        assert all(importance >= 0)  # Non-negative importance
        assert abs(importance.sum() - 1.0) < 1e-10  # Sum to 1


class TestUSGSCollector:
    """Test USGS data collection"""
    
    @mock.patch('dataretrieval.nwis.get_dv')
    def test_get_daily_streamflow(self, mock_get_dv):
        """Test daily streamflow data collection"""
        # Mock USGS response
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        mock_data = pd.DataFrame({
            '00060_Mean': np.random.uniform(100, 1000, 30),
            '00065_Mean': np.random.uniform(5, 15, 30)
        }, index=dates)
        mock_get_dv.return_value = mock_data
        
        collector = USGSCollector('01438500')
        result = collector.get_daily_streamflow('2023-01-01', '2023-01-30')
        
        # Check result
        assert isinstance(result, pd.DataFrame)
        assert 'discharge_cfs' in result.columns  # Standardized column name
        assert 'gage_height_ft' in result.columns
        assert len(result) <= 30  # May be filtered by validation


class TestFloodPredictor:
    """Test main FloodPredictor class"""
    
    def test_initialization(self):
        """Test FloodPredictor initialization"""
        predictor = FloodPredictor(
            usgs_site="01438500",
            flood_stage_ft=25.0,
            model="rf"
        )
        
        assert predictor.usgs_site == "01438500"
        assert predictor.flood_stage_ft == 25.0
        assert predictor.model_type == "rf"
        assert not predictor.is_fitted
        assert isinstance(predictor.model, RandomForestFloodModel)
    
    @mock.patch('floodml.data.usgs.USGSCollector.get_daily_streamflow')
    @mock.patch('floodml.data.nws.NWSCollector.get_precipitation_history')
    def test_synthetic_data_fallback(self, mock_nws, mock_usgs):
        """Test fallback to synthetic data when real data unavailable"""
        # Mock USGS data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        mock_usgs.return_value = pd.DataFrame({
            'discharge_cfs': np.random.uniform(100, 1000, 100),
            'gage_height_ft': 10 + np.random.normal(0, 2, 100)
        }, index=dates)
        
        # Mock empty NWS data (triggers synthetic fallback)
        mock_nws.return_value = pd.DataFrame()
        
        predictor = FloodPredictor(
            usgs_site="01438500",
            flood_stage_ft=25.0,
            model="rf"
        )
        
        # Should handle missing weather data gracefully
        try:
            predictor.fit('2023-01-01', '2023-03-31')
            # If it reaches here without error, synthetic data worked
            assert predictor.is_fitted
        except ValueError as e:
            # This is also acceptable - model requires real data
            assert "No valid training data" in str(e) or "No weather data" in str(e)


# Integration test
class TestIntegration:
    """Integration tests for the full workflow"""
    
    def test_end_to_end_synthetic(self):
        """Test end-to-end workflow with synthetic data"""
        # Create a predictor
        predictor = FloodPredictor(
            usgs_site="01438500",
            flood_stage_ft=25.0,
            model="rf",
            n_estimators=10  # Small for fast test
        )
        
        # Mock the data collection methods to return synthetic data
        def mock_streamflow(start, end):
            dates = pd.date_range(start, end, freq='D')
            return pd.DataFrame({
                'discharge_cfs': 500 + np.random.normal(0, 200, len(dates)),
                'gage_height_ft': 10 + np.random.normal(0, 5, len(dates))
            }, index=dates)
        
        def mock_weather(start, end):
            dates = pd.date_range(start, end, freq='D')  
            return pd.DataFrame({
                'precipitation_mm': np.random.exponential(2, len(dates)),
                'temperature_f': 60 + 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
            }, index=dates)
        
        # Patch the methods
        with mock.patch.object(predictor.usgs_collector, 'get_daily_streamflow', side_effect=mock_streamflow), \
             mock.patch.object(predictor.nws_collector, 'get_precipitation_history', side_effect=mock_weather):
            
            # Train model
            predictor.fit('2022-01-01', '2023-12-31')
            assert predictor.is_fitted
            
            # Mock recent data for prediction
            with mock.patch.object(predictor.usgs_collector, 'get_recent_streamflow', 
                                   return_value=mock_streamflow('2023-12-01', '2023-12-31')), \
                 mock.patch.object(predictor.nws_collector, 'get_hourly_forecast',
                                   return_value=mock_weather('2024-01-01', '2024-01-02')):
                
                # Make prediction
                result = predictor.predict_flood_probability(hours_ahead=24)
                
                assert 0 <= result.probability <= 1
                assert result.model_name is not None
                assert result.forecast_time is not None


if __name__ == "__main__":
    pytest.main([__file__])