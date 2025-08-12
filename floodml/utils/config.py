"""
Configuration management for FloodML
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import structlog

logger = structlog.get_logger()

# Default configuration
DEFAULT_CONFIG = {
    'user_agent': 'FloodML/0.1.0 (https://github.com/yourusername/floodml)',
    'logging': {
        'level': 'INFO',
        'format': 'json'
    },
    'data': {
        'cache_enabled': True,
        'cache_dir': '~/.floodml/cache',
        'request_timeout': 30,
        'retry_attempts': 3
    },
    'models': {
        'default_model': 'ensemble',
        'model_cache_dir': '~/.floodml/models',
        'feature_importance_threshold': 0.01
    },
    'api': {
        'usgs_base_url': 'https://waterservices.usgs.gov/nwis',
        'nws_base_url': 'https://api.weather.gov'
    }
}


class Config:
    """Configuration manager for FloodML"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Parameters
        ----------
        config_path : str, optional
            Path to configuration file
        """
        self._config = DEFAULT_CONFIG.copy()
        self.config_path = config_path
        
        # Load configuration from file if provided
        if config_path:
            self.load_from_file(config_path)
        
        # Load from environment variables
        self._load_from_env()
        
        # Expand paths
        self._expand_paths()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation, e.g., 'data.cache_dir')
        default : any, optional
            Default value if key not found
            
        Returns
        -------
        any
            Configuration value
        """
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        Set configuration value
        
        Parameters
        ----------
        key : str
            Configuration key (supports dot notation)
        value : any
            Value to set
        """
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file"""
        try:
            config_file = Path(config_path).expanduser()
            if config_file.exists():
                with open(config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        self._merge_config(file_config)
                logger.info("Configuration loaded from file", path=str(config_file))
            else:
                logger.info("Configuration file not found", path=str(config_file))
        except Exception as e:
            logger.error("Error loading configuration file", path=config_path, error=str(e))
    
    def save_to_file(self, config_path: str):
        """Save configuration to YAML file"""
        try:
            config_file = Path(config_path).expanduser()
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                yaml.dump(self._config, f, default_flow_style=False)
            
            logger.info("Configuration saved to file", path=str(config_file))
        except Exception as e:
            logger.error("Error saving configuration file", path=config_path, error=str(e))
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        env_mappings = {
            'FLOODML_USER_AGENT': 'user_agent',
            'FLOODML_LOG_LEVEL': 'logging.level',
            'FLOODML_CACHE_DIR': 'data.cache_dir',
            'FLOODML_MODEL_CACHE_DIR': 'models.model_cache_dir',
            'FLOODML_REQUEST_TIMEOUT': ('data.request_timeout', int),
        }
        
        for env_var, config_key in env_mappings.items():
            if isinstance(config_key, tuple):
                config_key, converter = config_key
            else:
                converter = str
            
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    self.set(config_key, converter(env_value))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid environment variable value", var=env_var, value=env_value, error=str(e))
    
    def _expand_paths(self):
        """Expand user paths in configuration"""
        path_keys = [
            'data.cache_dir',
            'models.model_cache_dir'
        ]
        
        for key in path_keys:
            value = self.get(key)
            if isinstance(value, str):
                expanded = str(Path(value).expanduser())
                self.set(key, expanded)
    
    def _merge_config(self, new_config: Dict[str, Any]):
        """Recursively merge new configuration with existing"""
        def merge_dict(base: Dict[str, Any], update: Dict[str, Any]):
            for key, value in update.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    merge_dict(base[key], value)
                else:
                    base[key] = value
        
        merge_dict(self._config, new_config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary"""
        return self._config.copy()
    
    def create_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.get('data.cache_dir'),
            self.get('models.model_cache_dir')
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    logger.debug("Created directory", path=dir_path)
                except Exception as e:
                    logger.error("Failed to create directory", path=dir_path, error=str(e))


# Global configuration instance
_global_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _global_config
    
    if _global_config is None:
        # Look for config file in common locations
        config_paths = [
            os.getenv('FLOODML_CONFIG'),
            '~/.floodml/config.yaml',
            './floodml.yaml',
            './config.yaml'
        ]
        
        config_file = None
        for path in config_paths:
            if path and Path(path).expanduser().exists():
                config_file = path
                break
        
        _global_config = Config(config_file)
        _global_config.create_directories()
    
    return _global_config


def set_config(config: Config):
    """Set global configuration instance"""
    global _global_config
    _global_config = config