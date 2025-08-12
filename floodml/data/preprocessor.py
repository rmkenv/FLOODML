"""
Data preprocessing and feature engineering for FloodML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Union
import structlog
from sklearn.preprocessing import StandardScaler, RobustScaler

logger = structlog.get_logger()


class DataPreprocessor:
    """
    Preprocesses and engineers features for flood prediction models
    
    Combines streamflow and weather data into ML-ready features with
    proper scaling and temporal feature engineering.
    """
    
    def __init__(self):
        """Initialize data preprocessor"""
        self.scaler = None
        self.feature_names = []
        self.target_scaler = None
        
    def create_flood_features(
        self,
        streamflow_data: pd.DataFrame,
        weather_data: pd.DataFrame,
        flood_stage: float,
        forecast_hours: int = 24
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features and targets for flood prediction
        
        Parameters
        ----------
        streamflow_data : pd.DataFrame
            Streamflow data with datetime index
        weather_data : pd.DataFrame  
            Weather/forecast data with datetime index
        flood_stage : float
            Flood stage threshold in feet
        forecast_hours : int, optional
            Hours ahead to predict (default: 24)
            
        Returns
        -------
        tuple
            (features_df, target_series)
        """
        logger.info("Creating flood prediction features", flood_stage=flood_stage, forecast_hours=forecast_hours)
        
        # Merge data on datetime index
        df = self._merge_data(streamflow_data, weather_data)
        
        if df.empty:
            logger.warning("No data available after merging")
            return pd.DataFrame(), pd.Series(dtype=float)
        
        # Engineer temporal features
        df = self._add_temporal_features(df)
        
        # Engineer hydrologic features
        df = self._add_hydrologic_features(df)
        
        # Engineer meteorologic features  
        df = self._add_meteorologic_features(df)
        
        # Create lag features
        df = self._add_lag_features(df)
        
        # Create moving averages and trends
        df = self._add_statistical_features(df)
        
        # Create flood target variable
        target = self._create_flood_target(df, flood_stage, forecast_hours)
        
        # Select feature columns
        feature_cols = self._select_feature_columns(df)
        features = df[feature_cols].copy()
        
        # Align features and target (remove rows where target is NaN due to forecasting)
        valid_idx = target.notna()
        features = features[valid_idx]
        target = target[valid_idx]
        
        # Remove rows with any NaN features
        complete_idx = features.notna().all(axis=1)
        features = features[complete_idx]
        target = target[complete_idx]
        
        self.feature_names = list(features.columns)
        
        logger.info(
            "Feature engineering complete",
            features=len(features.columns),
            samples=len(features),
            flood_events=target.sum() if len(target) > 0 else 0
        )
        
        return features, target
    
    def _merge_data(self, streamflow_data: pd.DataFrame, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Merge streamflow and weather data"""
        # Resample both datasets to hourly frequency
        streamflow_hourly = streamflow_data.resample('1H').mean()
        weather_hourly = weather_data.resample('1H').mean()
        
        # Forward fill weather data (forecasts are typically at lower frequency)
        weather_hourly = weather_hourly.ffill()
        
        # Merge on datetime index
        df = streamflow_hourly.join(weather_hourly, how='outer')
        
        # Forward fill streamflow data for small gaps
        for col in streamflow_hourly.columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill', limit=6)  # Max 6 hour gap
        
        return df
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def _add_hydrologic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add hydrologic-specific features"""
        df = df.copy()
        
        # Rate of change features
        if 'gage_height_ft' in df.columns:
            df['stage_rate_1h'] = df['gage_height_ft'].diff(1)
            df['stage_rate_3h'] = df['gage_height_ft'].diff(3) / 3
            df['stage_rate_6h'] = df['gage_height_ft'].diff(6) / 6
            df['stage_acceleration'] = df['stage_rate_1h'].diff(1)
        
        if 'discharge_cfs' in df.columns:
            df['discharge_rate_1h'] = df['discharge_cfs'].diff(1)
            df['discharge_rate_3h'] = df['discharge_cfs'].diff(3) / 3
            df['discharge_rate_6h'] = df['discharge_cfs'].diff(6) / 6
            
            # Log discharge (often more normal distribution)
            df['log_discharge'] = np.log1p(df['discharge_cfs'].fillna(0))
        
        return df
    
    def _add_meteorologic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add meteorologic features"""
        df = df.copy()
        
        # Precipitation accumulation features
        if 'precipitation_mm' in df.columns:
            df['precip_1h'] = df['precipitation_mm']
            df['precip_3h'] = df['precipitation_mm'].rolling(3).sum()
            df['precip_6h'] = df['precipitation_mm'].rolling(6).sum()
            df['precip_12h'] = df['precipitation_mm'].rolling(12).sum()
            df['precip_24h'] = df['precipitation_mm'].rolling(24).sum()
            df['precip_48h'] = df['precipitation_mm'].rolling(48).sum()
            df['precip_72h'] = df['precipitation_mm'].rolling(72).sum()
            
            # Precipitation intensity features
            df['precip_intensity_3h'] = df['precip_3h'] / 3
            df['precip_intensity_6h'] = df['precip_6h'] / 6
            df['precip_max_1h'] = df['precipitation_mm'].rolling(24).max()
            
            # Time since last significant precipitation
            significant_precip = df['precipitation_mm'] > 5  # 5mm threshold
            df['hours_since_precip'] = 0
            hours_counter = 0
            for i in range(len(df)):
                if significant_precip.iloc[i]:
                    hours_counter = 0
                else:
                    hours_counter += 1
                df['hours_since_precip'].iloc[i] = hours_counter
        
        # Temperature features
        if 'temperature_f' in df.columns:
            df['temp_c'] = (df['temperature_f'] - 32) * 5/9
            df['temp_trend_6h'] = df['temperature_f'].diff(6)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Add lagged features"""
        if lags is None:
            lags = [1, 3, 6, 12, 24, 48]  # 1h, 3h, 6h, 12h, 24h, 48h
        
        df = df.copy()
        
        # Lag key variables
        key_vars = ['gage_height_ft', 'discharge_cfs', 'precipitation_mm']
        
        for var in key_vars:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag_{lag}h'] = df[var].shift(lag)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame, windows: List[int] = None) -> pd.DataFrame:
        """Add statistical features (moving averages, std, etc.)"""
        if windows is None:
            windows = [6, 12, 24, 48, 72]  # Various time windows
        
        df = df.copy()
        
        # Key variables for statistical features
        key_vars = ['gage_height_ft', 'discharge_cfs']
        
        for var in key_vars:
            if var in df.columns:
                for window in windows:
                    # Moving statistics
                    df[f'{var}_ma_{window}h'] = df[var].rolling(window).mean()
                    df[f'{var}_std_{window}h'] = df[var].rolling(window).std()
                    df[f'{var}_min_{window}h'] = df[var].rolling(window).min()
                    df[f'{var}_max_{window}h'] = df[var].rolling(window).max()
                    df[f'{var}_range_{window}h'] = df[f'{var}_max_{window}h'] - df[f'{var}_min_{window}h']
                    
                    # Relative position within recent range
                    df[f'{var}_relative_{window}h'] = (
                        (df[var] - df[f'{var}_min_{window}h']) / 
                        (df[f'{var}_max_{window}h'] - df[f'{var}_min_{window}h'])
                    ).fillna(0)
        
        return df
    
    def _create_flood_target(self, df: pd.DataFrame, flood_stage: float, forecast_hours: int) -> pd.Series:
        """Create binary flood target variable"""
        if 'gage_height_ft' not in df.columns:
            logger.warning("No gage height data available for flood target")
            return pd.Series(dtype=float, index=df.index)
        
        # Create binary flood target (1 if flood stage exceeded in next N hours)
        future_stage = df['gage_height_ft'].shift(-forecast_hours)
        flood_target = (future_stage >= flood_stage).astype(int)
        
        return flood_target
    
    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select relevant feature columns"""
        # Exclude original data columns and intermediate calculations
        exclude_patterns = [
            'gage_height_ft',  # This is used for target creation
            'discharge_cfs',   # Use engineered versions instead
            'precipitation_mm', # Use engineered versions instead
            'temperature_f',   # Use temp_c instead
        ]
        
        exclude_exact = [
            'hour', 'day_of_week', 'day_of_year', 'month'  # Use cyclical versions
        ]
        
        feature_cols = []
        for col in df.columns:
            # Skip exact matches
            if col in exclude_exact:
                continue
                
            # Skip pattern matches
            if any(pattern in col for pattern in exclude_patterns if not any(suffix in col for suffix in ['_lag_', '_ma_', '_std_', '_min_', '_max_', '_range_', '_relative_', '_rate_', '_acceleration'])):
                continue
            
            feature_cols.append(col)
        
        return feature_cols
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, method: str = 'robust') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scale features using specified method
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        X_test : pd.DataFrame, optional
            Test features
        method : str, optional
            Scaling method ('standard', 'robust')
            
        Returns
        -------
        pd.DataFrame or tuple
            Scaled features
        """
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit scaler on training data
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        if X_test is not None:
            X_test_scaled = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names.copy()