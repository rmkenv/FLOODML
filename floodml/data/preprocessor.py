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
    Preprocesses and engineers features for flood prediction models.
    """

    def __init__(self):
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
        Create features and targets for flood prediction.
        """
        logger.info("Creating flood prediction features", 
                    flood_stage=flood_stage, forecast_hours=forecast_hours)

        df = self._merge_data(streamflow_data, weather_data)

        if df.empty:
            logger.warning("No data available after merging")
            return pd.DataFrame(), pd.Series(dtype=float)

        df = self._add_temporal_features(df)
        df = self._add_hydrologic_features(df)
        df = self._add_meteorologic_features(df)
        df = self._add_lag_features(df)
        df = self._add_statistical_features(df)

        target = self._create_flood_target(df, flood_stage, forecast_hours)

        feature_cols = self._select_feature_columns(df)
        features = df[feature_cols].copy()

        valid_idx = target.notna()
        features, target = features[valid_idx], target[valid_idx]

        complete_idx = features.notna().all(axis=1)
        features, target = features[complete_idx], target[complete_idx]

        self.feature_names = list(features.columns)

        logger.info(
            "Feature engineering complete",
            features=len(features.columns),
            samples=len(features),
            flood_events=int(target.sum())
        )

        return features, target

    def _merge_data(self, streamflow_data, weather_data):
        """Merge streamflow and weather data (hourly)."""
        # Use lowercase 'h' to avoid deprecation warnings
        streamflow_hourly = streamflow_data.resample('1h').mean()
        weather_hourly = weather_data.resample('1h').mean().ffill()

        df = streamflow_hourly.join(weather_hourly, how='outer')

        # Fill small streamflow gaps
        for col in streamflow_hourly.columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill', limit=6)

        return df

    def _add_temporal_features(self, df):
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        return df

    def _add_hydrologic_features(self, df):
        df = df.copy()
        if 'gage_height_ft' in df.columns:
            df['stage_rate_1h'] = df['gage_height_ft'].diff(1)
            df['stage_rate_3h'] = df['gage_height_ft'].diff(3) / 3
            df['stage_rate_6h'] = df['gage_height_ft'].diff(6) / 6
            df['stage_acceleration'] = df['stage_rate_1h'].diff(1)
        if 'discharge_cfs' in df.columns:
            df['discharge_rate_1h'] = df['discharge_cfs'].diff(1)
            df['discharge_rate_3h'] = df['discharge_cfs'].diff(3) / 3
            df['discharge_rate_6h'] = df['discharge_cfs'].diff(6) / 6
            df['log_discharge'] = np.log1p(df['discharge_cfs'].fillna(0))
        return df

    def _add_meteorologic_features(self, df):
        df = df.copy()
        if 'precipitation_mm' in df.columns:
            df['precip_1h'] = df['precipitation_mm']
            for h in [3, 6, 12, 24, 48, 72]:
                df[f'precip_{h}h'] = df['precipitation_mm'].rolling(h).sum()
            df['precip_intensity_3h'] = df['precip_3h'] / 3
            df['precip_intensity_6h'] = df['precip_6h'] / 6
            df['precip_max_1h'] = df['precipitation_mm'].rolling(24).max()

            # Time since significant precipitation (>5mm)
            sig_precip_mask = df['precipitation_mm'] > 5
            hours_since = []
            counter = 0
            for sig in sig_precip_mask:
                counter = 0 if sig else counter + 1
                hours_since.append(counter)
            df['hours_since_precip'] = hours_since

        if 'temperature_f' in df.columns:
            df['temp_c'] = (df['temperature_f'] - 32) * 5/9
            df['temp_trend_6h'] = df['temperature_f'].diff(6)

        return df

    def _add_lag_features(self, df, lags=None):
        if lags is None:
            lags = [1, 3, 6, 12, 24, 48]
        df = df.copy()
        for var in ['gage_height_ft', 'discharge_cfs', 'precipitation_mm']:
            if var in df.columns:
                for lag in lags:
                    df[f'{var}_lag_{lag}h'] = df[var].shift(lag)
        return df

    def _add_statistical_features(self, df, windows=None):
        if windows is None:
            windows = [6, 12, 24, 48, 72]
        df = df.copy()
        for var in ['gage_height_ft', 'discharge_cfs']:
            if var in df.columns:
                for w in windows:
                    df[f'{var}_ma_{w}h'] = df[var].rolling(w).mean()
                    df[f'{var}_std_{w}h'] = df[var].rolling(w).std()
                    df[f'{var}_min_{w}h'] = df[var].rolling(w).min()
                    df[f'{var}_max_{w}h'] = df[var].rolling(w).max()
                    df[f'{var}_range_{w}h'] = df[f'{var}_max_{w}h'] - df[f'{var}_min_{w}h']
                    denom = (df[f'{var}_max_{w}h'] - df[f'{var}_min_{w}h']).replace(0, np.nan)
                    df[f'{var}_relative_{w}h'] = ((df[var] - df[f'{var}_min_{w}h']) / denom).fillna(0)
        return df

    def _create_flood_target(self, df, flood_stage, forecast_hours):
        if 'gage_height_ft' not in df.columns or df['gage_height_ft'].isna().all():
            logger.warning("No usable gage height data for flood target")
            return pd.Series(dtype=float, index=df.index)
        future_stage = df['gage_height_ft'].shift(-forecast_hours)
        return (future_stage >= flood_stage).astype(int)

    def _select_feature_columns(self, df):
        exclude_patterns = [
            'gage_height_ft', 'discharge_cfs',
            'precipitation_mm', 'temperature_f'
        ]
        exclude_exact = ['hour', 'day_of_week', 'day_of_year', 'month']
        feature_cols = []
        for col in df.columns:
            if col in exclude_exact:
                continue
            if any(pattern in col for pattern in exclude_patterns
                   if not any(s in col for s in [
                       '_lag_', '_ma_', '_std_', '_min_', '_max_',
                       '_range_', '_relative_', '_rate_', '_acceleration'])
               ):
                continue
            feature_cols.append(col)
        return feature_cols

    def scale_features(self, X_train, X_test=None, method='robust'):
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")

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

    def get_feature_names(self):
        return self.feature_names.copy()
