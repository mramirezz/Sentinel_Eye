"""
Configuration management for Sentinel Eye.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration handler."""
    
    # Default configuration
    DEFAULT_CONFIG = {
        'video': {
            'input_path': 'data/',
            'output_path': 'outputs/',
            'target_resolution': [640, 480],
            'frame_skip': 1  # Process every N frames
        },
        'qc_score': {
            'weights': {
                'sharpness': 0.35,
                'occlusion': 0.25,
                'lighting': 0.20,
                'cleanliness': 0.20
            },
            'thresholds': {
                'excellent': 80,
                'good': 60,
                'warning': 40
            }
        },
        'stability': {
            'history_size': 30,
            'vibration_threshold': 5.0,
            'enable_self_healing': True
        },
        'optimization': {
            'use_gpu': True,
            'batch_size': 1,
            'enable_tensorrt': False,
            'enable_onnx': False
        },
        'logging': {
            'level': 'INFO',
            'save_plots': True,
            'save_videos': True
        },
        'roi': {
            'default': [100, 100, 440, 280],  # x, y, w, h
            'enable_adaptive': True
        }
    }
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML config file
        """
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
            self._deep_update(self.config, user_config)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict:
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path (e.g., 'video.target_resolution')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def save(self, output_path: str):
        """Save current configuration to file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
