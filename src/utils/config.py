"""
Configuration Management
Handles loading and validation of YAML configuration files
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

from .logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> DictConfig:
    """
    Load configuration from YAML file using OmegaConf
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        OmegaConf DictConfig object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    logger.info(f"Loading configuration from: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = OmegaConf.create(config_dict)
        logger.info(f"Configuration loaded successfully")
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config: DictConfig, save_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: OmegaConf DictConfig object
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving configuration to: {save_path}")
    
    with open(save_path, 'w') as f:
        OmegaConf.save(config, f)
    
    logger.info("Configuration saved successfully")


def merge_configs(base_config: DictConfig, override_config: DictConfig) -> DictConfig:
    """
    Merge two configurations with override taking precedence
    
    Args:
        base_config: Base configuration
        override_config: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = OmegaConf.merge(base_config, override_config)
    logger.info("Configurations merged successfully")
    return merged


def validate_config(config: DictConfig, required_keys: list) -> bool:
    """
    Validate that configuration contains required keys
    
    Args:
        config: Configuration to validate
        required_keys: List of required key paths (e.g., ['model.architecture', 'training.batch_size'])
        
    Returns:
        True if valid, raises ValueError otherwise
        
    Raises:
        ValueError: If required keys are missing
    """
    missing_keys = []
    
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        
        try:
            for key in keys:
                current = current[key]
        except (KeyError, AttributeError):
            missing_keys.append(key_path)
    
    if missing_keys:
        error_msg = f"Missing required configuration keys: {missing_keys}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Configuration validation passed")
    return True


class ConfigManager:
    """
    Centralized configuration management for DeepGuard-X
    """
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.configs: Dict[str, DictConfig] = {}
        logger.info(f"Initialized ConfigManager with directory: {self.config_dir}")
    
    def load(self, module: str, config_name: str) -> DictConfig:
        """
        Load configuration for a specific module
        
        Args:
            module: Module name (e.g., 'video', 'audio')
            config_name: Configuration file name without extension
            
        Returns:
            Loaded configuration
        """
        config_path = self.config_dir / module / f"{config_name}.yaml"
        config = load_config(str(config_path))
        
        # Cache configuration
        cache_key = f"{module}_{config_name}"
        self.configs[cache_key] = config
        
        return config
    
    def get(self, module: str, config_name: str) -> Optional[DictConfig]:
        """
        Get cached configuration
        
        Args:
            module: Module name
            config_name: Configuration name
            
        Returns:
            Cached configuration or None
        """
        cache_key = f"{module}_{config_name}"
        return self.configs.get(cache_key)
    
    def load_ensemble_config(self) -> DictConfig:
        """Load ensemble configuration"""
        return load_config(str(self.config_dir / "ensemble_config.yaml"))
    
    def load_realtime_config(self) -> DictConfig:
        """Load real-time configuration"""
        return load_config(str(self.config_dir / "realtime_config.yaml"))
