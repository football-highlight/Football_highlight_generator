"""
Configuration module for Football Highlights Generator
"""

from .config import config
from .config import VideoConfig, AudioConfig, ModelConfig, EventConfig, PathConfig, AppConfig

__version__ = "1.0.0"
__author__ = "Football Highlights Team"

__all__ = [
    'config',
    'VideoConfig',
    'AudioConfig', 
    'ModelConfig',
    'EventConfig',
    'PathConfig',
    'AppConfig',
    '__version__',
    '__author__'
]