"""
Main FloodPredictor class - high-level API for flood prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
import structlog

from ..data.usgs import USGSCollector
from ..data.nws import NWSCollector
from ..data.preprocessor import DataPreprocessor
from ..models.random_forest import RandomForestFloodModel
from ..models.lstm import LSTMFloodModel
from ..models.ensemble import EnsembleFloodModel
from ..models.base import FloodPredictionResult

logger = structlog.get_logger()


class FloodPredictor:
    """
    High-level API for flood prediction
    
    Combines data collection, preprocessing, and machine learning models
    to provide a simple interface for flood probability prediction.
    """
    
    def __init__(
        self,
        usgs_site: str,
        flood_stage_ft: float,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        model: str = "ensemble",
        **model_kwargs
    ):
        """
        Initialize FloodPredictor
        
        Parameters
        ----------
        usgs_site : str
            USGS site number (e.g., '01438500')
        flood_stage_ft : float
            Flood stage threshold in feet
        lat : float, optional
            Latitude for weather data (auto-detected if not provided)
        lon : float, optional
            Longitude for weather data (auto-detected if not provided)
        model : str, optional
            Model type ('rf', 'lstm', 'ensemble')
        **model_kwargs
            Additional arguments for the model
        """
        self.usgs_site = usgs_site
        self.flood_stage_ft = flood_stage_ft
        self.model_type = model.lower()
        
        # Initialize data collectors
        self.usgs_collector = USGSCollector(usgs_site)
        
        # Get coordinates if not provided
        if lat is None or lon is None:
            site_info = self.usgs_collector.get_site_info()
            self.lat = lat or site_info.get('dec_lat_va', 39.8283)
            self.lon = lon or site_info.get('dec_long_va', -98.5795)
        else:
            self.lat = lat
            self.lon = lon
        
        self.nws_collector = NWSCollector(self.lat, self.lon)
        self.preprocessor = DataPreprocessor()
        
        # Initialize model
        self.model = self._create_model(**model_kwargs)
        self.is_fitted = False
        
        logger.info(
            "Initialized FloodPredictor",
            site=usgs_site,
            flood_stage=flood_stage_ft,
            model_type=model,
            lat=self.lat,
            lon=self.lon
        )
    
    def _create_model(self, **kwargs):
        """Create the appropriate model based on model_type"""
        if self.model_type in ['rf', 'random_forest']:
            return RandomForestFloodModel(**kwargs)
        elif self.model_type == 'lstm':
            return LSTMFloodModel(**kwargs)
        elif self.model_type == 'ensemble':
            return EnsembleFloodModel(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(
        self,
        start_date: str,
        end_date: str,
        forecast_hours: int = 24,
        **fit_kwargs
    ) -> 'FloodPredictor':
        """
        Train the flood prediction model
        
        Parameters
        ----------
        start_date : str
            Training start date (YYYY-MM-DD)
        end_date : str
            Training end date (YYYY-MM-DD)
        forecast_hours : int, optional
            Hours ahead to predict (default: 24)
        **fit_kwargs
            Additional arguments for model fitting
            
        Returns
        -------
        FloodPredictor
            Fitted predictor instance
        """
        logger.info(
            "Training flood prediction model",
            start_date=start_date,
            end_date=end_date,
            forecast_hours=forecast_hours
        )
        
        # Collect training data
        logger.info("Collecting streamflow data...")
        streamflow_data = self.usgs_collector.get_daily_streamflow(start_date, end_date)
        
        logger.info("Collecting weather data...")
        weather_data = self.nws_collector.get_precipitation_history(start_date, end_date)
        
        if streamflow_data.empty:
            raise ValueError("No streamflow data available for training period")
        
        if weather_data.empty:
            logger.warning("No weather data available, using synthetic data")
            # Create synthetic precipitation data for training
            weather_data = self._create_synthetic_weather(streamflow_data.index)
        
        # Create features and targets
        logger.info("Engineering features...")
        features, targets = self.preprocessor.create_flood_features(
            streamflow_data, 
            weather_data,
            self.flood_stage_ft,
            forecast_hours
        )
        
        if features.empty or len(targets) == 0:
            raise ValueError("No valid training data after preprocessing")
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(features, targets, **fit_kwargs)
        self.is_fitted = True
        
        # Log training summary
        flood_events = targets.sum()
        total_samples = len(targets)
        
        logger.info(
            "Model training complete",
            total_samples=total_samples,
            flood_events=flood_events,
            flood_rate=f"{flood_events/total_samples:.1%}" if total_samples > 0 else "0%"
        )
        
        return self
    
    def predict_flood_probability(
        self,
        hours_ahead: int = 24,
        include_uncertainty: bool = True
    ) -> FloodPredictionResult:
        """
        Predict flood probability for the next N hours
        
        Parameters
        ----------
        hours_ahead : int, optional
            Hours ahead to predict (default: 24)
        include_uncertainty : bool, optional
            Whether to include uncertainty estimates
            
        Returns
        -------
        FloodPredictionResult
            Prediction result with probability and metadata
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        logger.info("Generating flood prediction", hours_ahead=hours_ahead)
        
        # Get recent data
        recent_streamflow = self.usgs_collector.get_recent_streamflow(days=30)
        forecast_weather = self.nws_collector.get_hourly_forecast(hours=hours_ahead + 24)
        
        if recent_streamflow.empty:
            raise ValueError("No recent streamflow data available")
        
        if forecast_weather.empty:
            logger.warning("No weather forecast available, using synthetic data")
            forecast_weather = self._create_synthetic_weather_forecast(hours_ahead + 24)
        
        # Prepare features
        features, _ = self.preprocessor.create_flood_features(
            recent_streamflow,
            forecast_weather,
            self.flood_stage_ft,
            hours_ahead
        )
        
        if features.empty:
            raise ValueError("No valid features for prediction")
        
        # Get the most recent feature vector
        latest_features = features.iloc[[-1]]
        
        # Make prediction
        probability = self.model.predict(latest_features)[0]
        
        # Get uncertainty estimates if available and requested
        confidence_low = None
        confidence_high = None
        
        if include_uncertainty and hasattr(self.model, 'predict_with_uncertainty'):
            try:
                prob_with_uncertainty = self.model.predict_with_uncertainty(latest_features)
                probability = prob_with_uncertainty['mean'][0]
                confidence_low = prob_with_uncertainty['lower'][0]
                confidence_high = prob_with_uncertainty['upper'][0]
            except:
                logger.warning("Could not compute uncertainty estimates")
        
        # Create result
        result = FloodPredictionResult(
            probability=probability,
            confidence_low=confidence_low,
            confidence_high=confidence_high,
            forecast_time=datetime.now() + timedelta(hours=hours_ahead),
            model_name=self.model.name,
            metadata={
                'usgs_site': self.usgs_site,
                'flood_stage_ft': self.flood_stage_ft,
                'hours_ahead': hours_ahead,
                'features_used': len(latest_features.columns),
                'model_type': self.model_type
            }
        )
        
        logger.info(
            "Prediction complete",
            probability=f"{probability:.1%}",
            hours_ahead=hours_ahead
        )
        
        return result
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """
        Get current hydrologic and meteorologic conditions
        
        Returns
        -------
        dict
            Current conditions summary
        """
        # Get latest streamflow data
        recent_flow = self.usgs_collector.get_recent_streamflow(days=1)
        current_weather = self.nws_collector.get_current_conditions()
        
        conditions = {
            'timestamp': datetime.now(),
            'usgs_site': self.usgs_site,
            'flood_stage_ft': self.flood_stage_ft,
        }
        
        # Add streamflow conditions
        if not recent_flow.empty:
            latest_flow = recent_flow.iloc[-1]
            conditions.update({
                'current_stage_ft': latest_flow.get('gage_height_ft'),
                'current_discharge_cfs': latest_flow.get('discharge_cfs'),
                'stage_above_flood': (
                    latest_flow.get('gage_height_ft', 0) - self.flood_stage_ft
                    if latest_flow.get('gage_height_ft') else None
                ),
                'data_timestamp': recent_flow.index[-1]
            })
        
        # Add weather conditions
        conditions.update({
            'current_weather': current_weather
        })
        
        return conditions
    
    def get_model_performance(self) -> Dict[str, Any]:
        """
        Get model performance metrics and information
        
        Returns
        -------
        dict
            Model performance and metadata
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}
        
        model_info = self.model.get_model_info()
        
        # Add FloodPredictor specific information
        model_info.update({
            'usgs_site': self.usgs_site,
            'flood_stage_ft': self.flood_stage_ft,
            'coordinates': (self.lat, self.lon)
        })
        
        return model_info
    
    def _create_synthetic_weather(self, date_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Create synthetic weather data for training when real data unavailable"""
        np.random.seed(42)  # For reproducibility
        
        # Generate realistic precipitation patterns
        precip = np.random.exponential(1.0, len(date_index))  # Exponential distribution
        temp_f = 60 + 30 * np.sin(2 * np.pi * date_index.dayofyear / 365) + np.random.normal(0, 5, len(date_index))
        
        weather_data = pd.DataFrame({
            'precipitation_mm': precip,
            'temperature_f': temp_f,
            'humidity_pct': np.random.uniform(30, 90, len(date_index))
        }, index=date_index)
        
        return weather_data
    
    def _create_synthetic_weather_forecast(self, hours: int) -> pd.DataFrame:
        """Create synthetic weather forecast when real forecast unavailable"""
        future_dates = pd.date_range(
            start=datetime.now(),
            periods=hours,
            freq='1H'
        )
        
        return self._create_synthetic_weather(future_dates)
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Save the model
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a previously trained model"""
        self.model = self.model.load(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")


# Convenience functions
def quick_flood_prediction(
    usgs_site: str,
    flood_stage_ft: float,
    hours_ahead: int = 24
) -> FloodPredictionResult:
    """
    Quick flood prediction using default settings
    
    Parameters
    ----------
    usgs_site : str
        USGS site number
    flood_stage_ft : float
        Flood stage threshold
    hours_ahead : int, optional
        Hours ahead to predict
        
    Returns
    -------
    FloodPredictionResult
        Prediction result
    """
    predictor = FloodPredictor(
        usgs_site=usgs_site,
        flood_stage_ft=flood_stage_ft,
        model="rf"  # Use fast random forest for quick predictions
    )
    
    # Train on last 2 years of data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    predictor.fit(start_date, end_date)
    
    return predictor.predict_flood_probability(hours_ahead=hours_ahead)