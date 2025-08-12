"""
National Weather Service (NWS) data collection using nwsapy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
import requests
import structlog
from nwsapy import api_connector

from ..utils.config import get_config
from .validation import DataValidator

logger = structlog.get_logger()


class NWSCollector:
    """
    Collects weather and forecast data from National Weather Service API
    
    Provides precipitation forecasts, temperature, and other meteorological
    variables needed for flood prediction.
    """
    
    def __init__(self, lat: float, lon: float, user_agent: Optional[str] = None):
        """
        Initialize NWS data collector
        
        Parameters
        ----------
        lat : float
            Latitude coordinate
        lon : float  
            Longitude coordinate
        user_agent : str, optional
            User agent string for API requests
        """
        self.lat = lat
        self.lon = lon
        self.user_agent = user_agent or get_config().get("user_agent", "FloodML/0.1.0")
        self.validator = DataValidator()
        self.base_url = "https://api.weather.gov"
        
        # Get grid information for this location
        self.grid_info = self._get_grid_info()
        
        logger.info("Initialized NWS collector", lat=lat, lon=lon)
    
    def _get_grid_info(self) -> Dict[str, Any]:
        """
        Get NWS grid information for the coordinates
        
        Returns
        -------
        dict
            Grid information including office and grid coordinates
        """
        try:
            url = f"{self.base_url}/points/{self.lat:.4f},{self.lon:.4f}"
            headers = {"User-Agent": self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            properties = data.get("properties", {})
            
            grid_info = {
                "office": properties.get("cwa"),
                "grid_x": properties.get("gridX"), 
                "grid_y": properties.get("gridY"),
                "forecast_url": properties.get("forecast"),
                "forecast_hourly_url": properties.get("forecastHourly"),
                "forecast_grid_data_url": properties.get("forecastGridData"),
            }
            
            logger.info("Retrieved grid info", lat=self.lat, lon=self.lon, office=grid_info["office"])
            return grid_info
            
        except Exception as e:
            logger.error("Failed to get grid info", lat=self.lat, lon=self.lon, error=str(e))
            return {}
    
    def get_hourly_forecast(self, hours: int = 168) -> pd.DataFrame:
        """
        Get hourly weather forecast
        
        Parameters
        ----------
        hours : int, optional
            Number of hours to forecast (default: 168 = 7 days)
            
        Returns
        -------
        pd.DataFrame
            Hourly forecast data with datetime index
        """
        try:
            logger.info("Fetching hourly forecast", lat=self.lat, lon=self.lon, hours=hours)
            
            # Use nwsapy for forecast
            forecast = api_connector.get_forecast_hourly(
                lat=self.lat,
                lon=self.lon, 
                user_agent=self.user_agent
            )
            
            forecast_data = []
            for i, period in enumerate(forecast.properties.periods):
                if i >= hours:
                    break
                    
                # Extract precipitation amount
                precip_amount = 0.0
                if hasattr(period, 'precipitationAmount') and period.precipitationAmount:
                    precip_amount = period.precipitationAmount.value or 0.0
                
                forecast_data.append({
                    'datetime': pd.to_datetime(period.startTime),
                    'temperature_f': period.temperature,
                    'humidity_pct': getattr(period, 'relativeHumidity', {}).get('value', None),
                    'precipitation_mm': precip_amount,
                    'wind_speed_mph': getattr(period, 'windSpeed', None),
                    'wind_direction': getattr(period, 'windDirection', None),
                })
            
            if not forecast_data:
                logger.warning("No forecast data retrieved")
                return pd.DataFrame()
            
            df = pd.DataFrame(forecast_data)
            df.set_index('datetime', inplace=True)
            
            # Validate data
            df = self.validator.validate_weather(df)
            
            logger.info(
                "Successfully retrieved forecast", 
                lat=self.lat, 
                lon=self.lon,
                periods=len(df)
            )
            
            return df
            
        except Exception as e:
            logger.error("Failed to get hourly forecast", lat=self.lat, lon=self.lon, error=str(e))
            return pd.DataFrame()
    
    def get_quantitative_precipitation_forecast(self, hours: int = 168) -> pd.DataFrame:
        """
        Get quantitative precipitation forecast (QPF) from grid data
        
        Parameters
        ----------
        hours : int, optional
            Number of hours to forecast
            
        Returns
        -------
        pd.DataFrame
            QPF data with datetime index
        """
        if not self.grid_info.get("forecast_grid_data_url"):
            logger.warning("No grid data URL available")
            return pd.DataFrame()
        
        try:
            logger.info("Fetching QPF data", lat=self.lat, lon=self.lon)
            
            # Get QPF from grid data
            url = f"{self.grid_info['forecast_grid_data_url']}/quantitativePrecipitation"
            headers = {"User-Agent": self.user_agent}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            values = data.get("properties", {}).get("values", [])
            
            qpf_data = []
            for value in values:
                valid_time = value.get("validTime", "")
                if "/" in valid_time:
                    start_time = valid_time.split("/")[0]
                    try:
                        dt = pd.to_datetime(start_time)
                        precip_value = value.get("value")
                        if precip_value is not None:
                            qpf_data.append({
                                'datetime': dt,
                                'qpf_mm': precip_value
                            })
                    except:
                        continue
            
            if not qpf_data:
                logger.warning("No QPF data found")
                return pd.DataFrame()
            
            df = pd.DataFrame(qpf_data)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
            # Limit to requested hours
            end_time = datetime.now() + timedelta(hours=hours)
            df = df[df.index <= end_time]
            
            logger.info("Successfully retrieved QPF", records=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to get QPF", lat=self.lat, lon=self.lon, error=str(e))
            return pd.DataFrame()
    
    def get_precipitation_history(
        self, 
        start_date: str, 
        end_date: str,
        source: str = "gridded"
    ) -> pd.DataFrame:
        """
        Get historical precipitation data
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
        source : str, optional
            Data source ('gridded' or 'station')
            
        Returns
        -------
        pd.DataFrame
            Historical precipitation data
        """
        try:
            logger.info(
                "Fetching precipitation history", 
                lat=self.lat, 
                lon=self.lon,
                start=start_date,
                end=end_date
            )
            
            if source == "gridded":
                return self._get_gridded_precipitation_history(start_date, end_date)
            else:
                return self._get_station_precipitation_history(start_date, end_date)
                
        except Exception as e:
            logger.error("Failed to get precipitation history", error=str(e))
            return pd.DataFrame()
    
    def _get_gridded_precipitation_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get gridded precipitation from NOAA/NWS"""
        # This would typically require additional APIs like NOAA Climate Data Online
        # For now, return synthetic data as placeholder
        logger.warning("Gridded precipitation history not yet implemented")
        
        # Generate synthetic precipitation data for demonstration
        date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
        synthetic_precip = np.random.exponential(1.0, len(date_range))  # mm/hour
        
        df = pd.DataFrame({
            'precipitation_mm': synthetic_precip
        }, index=date_range)
        
        return df
    
    def _get_station_precipitation_history(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get station-based precipitation data"""
        # This would require integration with additional APIs
        logger.warning("Station precipitation history not yet implemented")
        return pd.DataFrame()
    
    def get_current_conditions(self) -> Dict[str, Any]:
        """
        Get current weather conditions
        
        Returns
        -------
        dict
            Current weather conditions
        """
        try:
            # Get current conditions from nearest weather station
            stations_url = f"{self.base_url}/points/{self.lat:.4f},{self.lon:.4f}/stations"
            headers = {"User-Agent": self.user_agent}
            
            response = requests.get(stations_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            stations_data = response.json()
            stations = stations_data.get("features", [])
            
            if not stations:
                logger.warning("No weather stations found nearby")
                return {}
            
            # Use first station
            station_id = stations[0]["properties"]["stationIdentifier"]
            
            # Get current observations
            obs_url = f"{self.base_url}/stations/{station_id}/observations/latest"
            obs_response = requests.get(obs_url, headers=headers, timeout=30)
            obs_response.raise_for_status()
            
            obs_data = obs_response.json()
            properties = obs_data.get("properties", {})
            
            conditions = {
                "timestamp": properties.get("timestamp"),
                "temperature_c": properties.get("temperature", {}).get("value"),
                "humidity_pct": properties.get("relativeHumidity", {}).get("value"), 
                "pressure_pa": properties.get("barometricPressure", {}).get("value"),
                "wind_speed_ms": properties.get("windSpeed", {}).get("value"),
                "precipitation_mm": properties.get("precipitationLastHour", {}).get("value"),
            }
            
            logger.info("Retrieved current conditions", station=station_id)
            return conditions
            
        except Exception as e:
            logger.error("Failed to get current conditions", error=str(e))
            return {}
    
    def get_alerts(self, active_only: bool = True) -> pd.DataFrame:
        """
        Get weather alerts for the area
        
        Parameters
        ----------
        active_only : bool, optional
            Only return active alerts (default: True)
            
        Returns
        -------
        pd.DataFrame
            Weather alerts data
        """
        try:
            url = f"{self.base_url}/alerts"
            params = {
                "point": f"{self.lat},{self.lon}",
                "status": "actual" if active_only else None
            }
            headers = {"User-Agent": self.user_agent}
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            features = data.get("features", [])
            
            if not features:
                return pd.DataFrame()
            
            alerts_data = []
            for feature in features:
                props = feature.get("properties", {})
                alerts_data.append({
                    'id': props.get('id'),
                    'event': props.get('event'),
                    'severity': props.get('severity'),
                    'certainty': props.get('certainty'),
                    'urgency': props.get('urgency'),
                    'headline': props.get('headline'),
                    'description': props.get('description'),
                    'onset': pd.to_datetime(props.get('onset')),
                    'expires': pd.to_datetime(props.get('expires')),
                })
            
            df = pd.DataFrame(alerts_data)
            logger.info("Retrieved weather alerts", count=len(df))
            return df
            
        except Exception as e:
            logger.error("Failed to get weather alerts", error=str(e))
            return pd.DataFrame()