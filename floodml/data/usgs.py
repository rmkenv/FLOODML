"""
USGS data collection using dataretrieval and hydrofunctions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import dataretrieval.nwis as nwis
import hydrofunctions as hf
import structlog

from ..utils.config import get_config
from .validation import DataValidator

logger = structlog.get_logger()


class USGSCollector:
    """
    Collects streamflow data from USGS National Water Information System (NWIS)
    
    Combines dataretrieval for historical daily values and hydrofunctions 
    for real-time instantaneous values.
    """
    
    def __init__(self, site: str, user_agent: Optional[str] = None):
        """
        Initialize USGS data collector
        
        Parameters
        ----------
        site : str
            USGS site number (e.g., '01438500')
        user_agent : str, optional
            User agent string for API requests
        """
        self.site = site
        self.user_agent = user_agent or get_config().get("user_agent", "FloodML/0.1.0")
        self.validator = DataValidator()
        
        logger.info("Initialized USGS collector", site=site)
    
    def get_site_info(self) -> Dict[str, Any]:
        """
        Get site information and metadata
        
        Returns
        -------
        dict
            Site information including lat/lon, drainage area, etc.
        """
        try:
            site_info = nwis.get_info(sites=self.site)
            if not site_info.empty:
                info = site_info.iloc[0].to_dict()
                logger.info("Retrieved site info", site=self.site)
                return info
            else:
                logger.warning("No site info found", site=self.site)
                return {}
        except Exception as e:
            logger.error("Failed to get site info", site=self.site, error=str(e))
            return {}
    
    def get_daily_streamflow(
        self, 
        start_date: str, 
        end_date: str,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get daily streamflow values
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str  
            End date in YYYY-MM-DD format
        parameters : list, optional
            Parameter codes to retrieve (default: ['00060', '00065'])
            
        Returns
        -------
        pd.DataFrame
            Daily streamflow data with datetime index
        """
        if parameters is None:
            parameters = ['00060', '00065']  # Discharge and gage height
        
        try:
            logger.info(
                "Fetching daily streamflow", 
                site=self.site, 
                start=start_date, 
                end=end_date,
                parameters=parameters
            )
            
            # Get daily values using dataretrieval
            df = nwis.get_dv(
                sites=self.site,
                parameterCd=parameters, 
                start=start_date,
                end=end_date
            )
            
            if df.empty:
                logger.warning("No daily data found", site=self.site)
                return pd.DataFrame()
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Validate data quality
            df = self.validator.validate_streamflow(df)
            
            logger.info(
                "Successfully retrieved daily data", 
                site=self.site,
                records=len(df),
                date_range=(df.index.min(), df.index.max())
            )
            
            return df
            
        except Exception as e:
            logger.error("Failed to get daily streamflow", site=self.site, error=str(e))
            return pd.DataFrame()
    
    def get_instantaneous_streamflow(
        self,
        start_date: str,
        end_date: str,
        parameters: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get instantaneous (15-minute) streamflow values using hydrofunctions
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format  
        parameters : list, optional
            Parameter codes to retrieve
            
        Returns
        -------
        pd.DataFrame
            Instantaneous streamflow data with datetime index
        """
        if parameters is None:
            parameters = ['00060', '00065']
            
        try:
            logger.info(
                "Fetching instantaneous streamflow",
                site=self.site,
                start=start_date, 
                end=end_date
            )
            
            # Use hydrofunctions for instantaneous values
            station = hf.NWIS(
                self.site,
                service='iv',
                start_date=start_date,
                end_date=end_date,
                parameterCd=','.join(parameters)
            )
            
            if not station.ok:
                logger.warning("Failed to retrieve instantaneous data", site=self.site)
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = station.df()
            
            if df.empty:
                logger.warning("No instantaneous data found", site=self.site)  
                return pd.DataFrame()
            
            # Reset index to make datetime a column, then set as index
            df = df.reset_index()
            if 'datetime' in df.columns:
                df.set_index('datetime', inplace=True)
            elif df.index.name is None:
                # If no datetime column, assume index is already datetime
                df.index.name = 'datetime'
            
            # Standardize column names
            df = self._standardize_columns(df)
            
            # Resample to hourly to reduce data size
            df = df.resample('1H').mean()
            
            # Validate data
            df = self.validator.validate_streamflow(df)
            
            logger.info(
                "Successfully retrieved instantaneous data",
                site=self.site,
                records=len(df),
                date_range=(df.index.min(), df.index.max())
            )
            
            return df
            
        except Exception as e:
            logger.error("Failed to get instantaneous streamflow", site=self.site, error=str(e))
            return pd.DataFrame()
    
    def get_recent_streamflow(self, days: int = 30) -> pd.DataFrame:
        """
        Get recent streamflow data for the last N days
        
        Parameters
        ----------
        days : int, optional
            Number of days to retrieve (default: 30)
            
        Returns
        -------
        pd.DataFrame
            Recent streamflow data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.get_instantaneous_streamflow(start_date, end_date)
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names for consistent API
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw dataframe from USGS
            
        Returns
        -------
        pd.DataFrame
            DataFrame with standardized column names
        """
        # Common USGS parameter code mappings
        column_mapping = {
            '00060_Mean': 'discharge_cfs',
            '00065_Mean': 'gage_height_ft', 
            '00010_Mean': 'water_temp_c',
            '00060': 'discharge_cfs',
            '00065': 'gage_height_ft',
            '00010': 'water_temp_c',
        }
        
        # Apply mapping
        df = df.rename(columns=column_mapping)
        
        # Handle any remaining parameter codes
        for col in df.columns:
            if col.startswith('00060'):
                df = df.rename(columns={col: 'discharge_cfs'})
            elif col.startswith('00065'):
                df = df.rename(columns={col: 'gage_height_ft'})
            elif col.startswith('00010'):
                df = df.rename(columns={col: 'water_temp_c'})
        
        return df
    
    def get_flood_stage_info(self) -> Dict[str, float]:
        """
        Attempt to get flood stage information for the site
        
        Returns
        -------
        dict
            Flood stage thresholds if available
        """
        try:
            # This would need to be implemented with additional USGS services
            # For now, return empty dict - users should provide thresholds
            logger.info("Flood stage info not automatically available", site=self.site)
            return {}
        except Exception as e:
            logger.error("Failed to get flood stage info", site=self.site, error=str(e))
            return {}
    
    def get_parameter_info(self) -> pd.DataFrame:
        """
        Get information about available parameters at the site
        
        Returns
        -------
        pd.DataFrame
            Available parameters and their descriptions
        """
        try:
            info = nwis.get_info(sites=self.site, parameterCd='all')
            logger.info("Retrieved parameter info", site=self.site, params=len(info))
            return info
        except Exception as e:
            logger.error("Failed to get parameter info", site=self.site, error=str(e))
            return pd.DataFrame()