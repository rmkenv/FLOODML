"""
Data validation and quality control for FloodML
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import structlog

logger = structlog.get_logger()


class DataValidator:
    """
    Validates and cleans data from various sources.
    Implements quality control checks specific to hydrologic and meteorologic data.
    """

    def __init__(self):
        self.validation_stats = {
            'total_records': 0,
            'removed_duplicates': 0,
            'removed_outliers': 0,
            'filled_missing': 0,
        }

    def validate_streamflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate streamflow data from USGS."""
        if df.empty:
            return df

        logger.info("Validating streamflow data", records=len(df))
        original_count = len(df)

        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = original_count - len(df)

        if 'discharge_cfs' in df.columns:
            df = self._validate_discharge(df)
        if 'gage_height_ft' in df.columns:
            df = self._validate_gage_height(df)

        df = df.sort_index()

        self.validation_stats['total_records'] += original_count
        self.validation_stats['removed_duplicates'] += duplicates_removed

        logger.info(
            "Streamflow validation complete",
            original=original_count,
            final=len(df),
            removed=original_count - len(df)
        )

        return df

    def validate_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate weather/forecast data."""
        if df.empty:
            return df

        logger.info("Validating weather data", records=len(df))
        original_count = len(df)
        df = df[~df.index.duplicated(keep='first')]

        if 'temperature_f' in df.columns:
            df = self._validate_temperature(df)
        if 'precipitation_mm' in df.columns:
            df = self._validate_precipitation(df)
        if 'humidity_pct' in df.columns:
            df = self._validate_humidity(df)

        df = df.sort_index()

        logger.info(
            "Weather validation complete",
            original=original_count,
            final=len(df)
        )
        return df

    def _validate_discharge(self, df: pd.DataFrame) -> pd.DataFrame:
        col = 'discharge_cfs'
        # Remove negative values
        invalid_mask = pd.to_numeric(df[col], errors='coerce') < 0
        df.loc[invalid_mask, col] = np.nan

        series = pd.to_numeric(df[col], errors='coerce')
        if series.size > 0:
            lower_bound = series.quantile(0.001)
            upper_bound = series.quantile(0.999)
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            df.loc[outlier_mask, col] = np.nan
            self.validation_stats['removed_outliers'] += outlier_count
            if outlier_count > 0:
                logger.info(f"Removed {outlier_count} discharge outliers")

        filled_before = df[col].isna().sum()
        df[col] = df[col].interpolate(method='linear', limit=3)
        filled_after = df[col].isna().sum()
        self.validation_stats['filled_missing'] += (filled_before - filled_after)
        return df

    def _validate_gage_height(self, df: pd.DataFrame) -> pd.DataFrame:
        col = 'gage_height_ft'
        series = pd.to_numeric(df[col], errors='coerce')
        if series.size > 0:
            lower_bound = series.quantile(0.001)
            upper_bound = series.quantile(0.999)
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_count = int(outlier_mask.sum())
            df.loc[outlier_mask, col] = np.nan
            self.validation_stats['removed_outliers'] += outlier_count
        filled_before = df[col].isna().sum()
        df[col] = df[col].interpolate(method='linear', limit=3)
        filled_after = df[col].isna().sum()
        self.validation_stats['filled_missing'] += (filled_before - filled_after)
        return df

    def _validate_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        col = 'temperature_f'
        invalid_mask = (df[col] < -100) | (df[col] > 150)
        df.loc[invalid_mask, col] = np.nan
        return df

    def _validate_precipitation(self, df: pd.DataFrame) -> pd.DataFrame:
        col = 'precipitation_mm'
        invalid_mask = df[col] < 0
        df.loc[invalid_mask, col] = np.nan
        series = df[col]
        if series.size > 0:
            extreme_mask = series > 200
            extreme_count = int(extreme_mask.sum())
            if extreme_count > 0:
                logger.warning(f"Found {extreme_count} extreme precipitation values (>200mm/hr)")
        # Fill missing with zero
        filled_before = df[col].isna().sum()
        df[col] = df[col].fillna(0)
        filled_after = df[col].isna().sum()
        self.validation_stats['filled_missing'] += (filled_before - filled_after)
        return df

    def _validate_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        col = 'humidity_pct'
        invalid_mask = (df[col] < 0) | (df[col] > 100)
        df.loc[invalid_mask, col] = np.nan
        return df

    def check_data_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, float]:
        stats = {}
        for col in required_columns:
            if col in df.columns and len(df) > 0:
                completeness = (1 - df[col].isna().sum() / len(df)) * 100
                stats[col] = round(float(completeness), 2)
            else:
                stats[col] = 0.0
        return stats

    def detect_anomalies(self, df: pd.DataFrame, column: str, method: str = 'zscore') -> pd.Series:
        if column not in df.columns or len(df[column].dropna()) == 0:
            return pd.Series(False, index=df.index)

        data = pd.to_numeric(df[column], errors='coerce').dropna()

        if method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = z_scores > 3
        elif method == 'iqr':
            Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
            IQR = Q3 - Q1
            anomalies = (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")

        result = pd.Series(False, index=df.index)
        result.loc[anomalies.index] = anomalies
        return result

    def get_validation_summary(self) -> Dict[str, int]:
        return self.validation_stats.copy()

    def reset_stats(self):
        self.validation_stats = {
            'total_records': 0,
            'removed_duplicates': 0,
            'removed_outliers': 0,
            'filled_missing': 0,
        }
