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
    Validates and cleans data from various sources
    
    Implements quality control checks specific to hydrologic and meteorologic data.
    """
    
    def __init__(self):
        """Initialize data validator"""
        self.validation_stats = {
            'total_records': 0,
            'removed_duplicates': 0,
            'removed_outliers': 0,
            'filled_missing': 0,
        }
    
    def validate_streamflow(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate streamflow data from USGS
        
        Parameters
        ----------
        df : pd.DataFrame
            Streamflow data with datetime index
            
        Returns
        -------
        pd.DataFrame
            Validated and cleaned streamflow data
        """
        if df.empty:
            return df
        
        logger.info("Validating streamflow data", records=len(df))
        original_count = len(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        duplicates_removed = original_count - len(df)
        
        # Validate discharge values (if present)
        if 'discharge_cfs' in df.columns:
            df = self._validate_discharge(df)
        
        # Validate gage height values (if present)
        if 'gage_height_ft' in df.columns:
            df = self._validate_gage_height(df)
        
        # Sort by datetime
        df = df.sort_index()
        
        # Update stats
        self.validation_stats['total_records'] += original_count
        self.validation_stats['removed_duplicates'] += duplicates_removed
        
        logger.info(
            "Streamflow validation complete",
            original=original_count,
            final=len(df),
            removed=original_count-len(df)
        )
        
        return df
    
    def validate_weather(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate weather/forecast data from NWS
        
        Parameters
        ----------
        df : pd.DataFrame
            Weather data with datetime index
            
        Returns
        -------
        pd.DataFrame
            Validated weather data
        """
        if df.empty:
            return df
        
        logger.info("Validating weather data", records=len(df))
        original_count = len(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Validate temperature
        if 'temperature_f' in df.columns:
            df = self._validate_temperature(df)
        
        # Validate precipitation
        if 'precipitation_mm' in df.columns:
            df = self._validate_precipitation(df)
        
        # Validate humidity
        if 'humidity_pct' in df.columns:
            df = self._validate_humidity(df)
        
        # Sort by datetime
        df = df.sort_index()
        
        logger.info(
            "Weather validation complete", 
            original=original_count,
            final=len(df)
        )
        
        return df
    
    def _validate_discharge(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate discharge values"""
        discharge_col = 'discharge_cfs'
        
        # Remove negative discharge values
        invalid_mask = df[discharge_col] < 0
        df.loc[invalid_mask, discharge_col] = np.nan
        
        # Remove extreme outliers (> 99.9th percentile or < 0.1st percentile)
        if not df[discharge_col].empty:
            lower_bound = df[discharge_col].quantile(0.001)
            upper_bound = df[discharge_col].quantile(0.999)
            
            outlier_mask = (df[discharge_col] < lower_bound) | (df[discharge_col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            df.loc[outlier_mask, discharge_col] = np.nan
            self.validation_stats['removed_outliers'] += outlier_count
            
            if outlier_count > 0:
                logger.info(f"Removed {outlier_count} discharge outliers")
        
        # Fill small gaps (up to 3 hours) with interpolation
        df[discharge_col] = df[discharge_col].interpolate(method='linear', limit=3)
        
        return df
    
    def _validate_gage_height(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate gage height values"""
        height_col = 'gage_height_ft'
        
        # Remove extreme outliers
        if not df[height_col].empty:
            lower_bound = df[height_col].quantile(0.001)
            upper_bound = df[height_col].quantile(0.999)
            
            outlier_mask = (df[height_col] < lower_bound) | (df[height_col] > upper_bound)
            outlier_count = outlier_mask.sum()
            
            df.loc[outlier_mask, height_col] = np.nan
            self.validation_stats['removed_outliers'] += outlier_count
        
        # Fill small gaps with interpolation
        df[height_col] = df[height_col].interpolate(method='linear', limit=3)
        
        return df
    
    def _validate_temperature(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate temperature values"""
        temp_col = 'temperature_f'
        
        # Remove physically impossible temperatures (outside -100 to 150°F)
        invalid_mask = (df[temp_col] < -100) | (df[temp_col] > 150)
        df.loc[invalid_mask, temp_col] = np.nan
        
        return df
    
    def _validate_precipitation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate precipitation values"""
        precip_col = 'precipitation_mm'
        
        # Remove negative precipitation
        invalid_mask = df[precip_col] < 0
        df.loc[invalid_mask, precip_col] = np.nan
        
        # Remove extreme outliers (> 200mm/hour is very rare)
        if not df[precip_col].empty:
            extreme_mask = df[precip_col] > 200
            extreme_count = extreme_mask.sum()
            
            if extreme_count > 0:
                logger.warning(f"Found {extreme_count} extreme precipitation values")
                # Don't automatically remove - could be real extreme events
                # Just log for review
        
        # Fill missing with 0 (assume no precipitation if not reported)
        df[precip_col] = df[precip_col].fillna(0)
        
        return df
    
    def _validate_humidity(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate humidity values"""
        humidity_col = 'humidity_pct'
        
        # Ensure humidity is between 0 and 100%
        invalid_mask = (df[humidity_col] < 0) | (df[humidity_col] > 100)
        df.loc[invalid_mask, humidity_col] = np.nan
        
        return df
    
    def check_data_completeness(self, df: pd.DataFrame, required_columns: List[str]) -> Dict[str, float]:
        """
        Check data completeness for required columns
        
        Parameters
        ----------
        df : pd.DataFrame
            Data to check
        required_columns : list
            List of required column names
            
        Returns
        -------
        dict
            Completeness statistics for each column
        """
        stats = {}
        
        for col in required_columns:
            if col in df.columns:
                completeness = (1 - df[col].isna().sum() / len(df)) * 100
                stats[col] = completeness
            else:
                stats[col] = 0.0
                
        return stats
    
    def detect_anomalies(self, df: pd.DataFrame, column: str, method: str = 'zscore') -> pd.Series:
        """
        Detect anomalies in a data series
        
        Parameters
        ----------
        df : pd.DataFrame
            Data containing the column to analyze
        column : str
            Column name to analyze
        method : str, optional
            Anomaly detection method ('zscore', 'iqr')
            
        Returns
        -------
        pd.Series
            Boolean series indicating anomalies
        """
        if column not in df.columns or df[column].empty:
            return pd.Series(False, index=df.index)
        
        data = df[column].dropna()
        
        if method == 'zscore':
            z_scores = np.abs((data - data.mean()) / data.std())
            anomalies = z_scores > 3
            
        elif method == 'iqr':
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            anomalies = (data < lower_bound) | (data > upper_bound)
            
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        # Align with original dataframe index
        result = pd.Series(False, index=df.index)
        result.loc[anomalies.index] = anomalies
        
        return result
    
    def get_validation_summary(self) -> Dict[str, int]:
        """
        Get summary of validation actions performed
        
        Returns
        -------
        dict
            Summary statistics of validation actions
        """
        return self.validation_stats.copy()
    
    def reset_stats(self):
        """Reset validation statistics"""
        self.validation_stats = {
            'total_records': 0,
            'removed_duplicates': 0,
            'removed_outliers': 0,
            'filled_missing': 0,
        }