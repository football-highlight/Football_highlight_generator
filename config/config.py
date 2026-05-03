"""
Main configuration file for Football Highlights Generator
UPDATED VERSION - Fixed property setter issue
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Video configuration
@dataclass
class VideoConfig:
    # CRITICAL: These dimensions must match what your model expects
    FRAME_RATE: int = 25
    FRAME_WIDTH: int = 160    # MUST be 160 for the model
    FRAME_HEIGHT: int = 90    # MUST be 90 for the model
    CLIP_DURATION: int = 1    # 1 second clips
    
    # Video processing parameters
    NUM_FRAMES: int = 16  # For model input
    MEAN: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    STD: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Calculate frames per clip based on duration
    @property
    def FRAMES_PER_CLIP(self) -> int:
        return int(self.FRAME_RATE * self.CLIP_DURATION)
    
    # Video-specific buffers for clip creation
    CLIP_BUFFER_BEFORE: int = 2
    CLIP_BUFFER_AFTER: int = 5

# Audio configuration  
@dataclass
class AudioConfig:
    SAMPLE_RATE: int = 16000
    LANGUAGE: str = "en"
    WHISPER_MODEL: str = "base"
    MFCC_N_COEFFS: int = 13
    # Audio-specific buffers for keyword detection timing
    KEYWORD_BUFFER_BEFORE: int = 5
    KEYWORD_BUFFER_AFTER: int = 10

# Model configuration
@dataclass
class ModelConfig:
    CNN_INPUT_SHAPE: Tuple[int, int, int, int] = (16, 90, 160, 3)
    CNN_BATCH_SIZE: int = 2
    CNN_LEARNING_RATE: float = 1e-3
    CNN_DROPOUT_RATE: float = 0.3
    NUM_EPOCHS: int = 10
    PATIENCE: int = 5
    MIN_DELTA: float = 0.001
    OPTIMIZER: str = "adam"
    WEIGHT_DECAY: float = 1e-5
    SCHEDULER: str = "reduce_on_plateau"
    GRADIENT_CLIP: float = 1.0
    
    # Model paths
    CHECKPOINT_PATH: Optional[str] = None
    PRETRAINED_PATH: Optional[str] = None

# Event configuration
@dataclass
class EventConfig:
    KEYWORDS: List[str] = field(default_factory=lambda: [
        "goal", "red card", "yellow card", "foul", "penalty",
        "free kick", "corner", "offside", "save", "miss"
    ])
    
    # Event weights for importance scoring
    EVENT_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "goal": 1.0,
        "penalty": 0.9,
        "red card": 0.8,
        "yellow card": 0.7,
        "save": 0.6,
        "miss": 0.5,
        "free kick": 0.4,
        "corner": 0.4,
        "foul": 0.3,
        "offside": 0.2
    })
    
    CONFIDENCE_THRESHOLD: float = 0.7
    
    # Event detection parameters
    MIN_EVENT_DURATION: float = 3.0  # Minimum event duration in seconds
    MAX_EVENT_DURATION: float = 15.0  # Maximum event duration in seconds
    MERGE_THRESHOLD: float = 2.0  # Merge events within this time gap (seconds)
    
    # Buffers for event highlighting
    EVENT_BUFFER_BEFORE: int = 5
    EVENT_BUFFER_AFTER: int = 10
    
    # Highlight generation parameters
    MAX_HIGHLIGHTS_DURATION: int = 600  # Maximum highlights duration in seconds (10 minutes)
    MIN_HIGHLIGHT_DURATION: int = 3  # Minimum highlight clip duration
    MAX_HIGHLIGHT_DURATION: int = 30  # Maximum highlight clip duration
    
    Merge_Window: float = 2.0

# Path configuration
@dataclass
class PathConfig:
    # Get project root (two levels up from config directory)
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    
    # Data directories
    RAW_VIDEO_DIR: Path = PROJECT_ROOT / "data" / "raw_videos"
    PROCESSED_VIDEO_DIR: Path = PROJECT_ROOT / "data" / "processed_videos"
    HIGHLIGHTS_DIR: Path = PROJECT_ROOT / "data" / "highlights"
    ANNOTATIONS_DIR: Path = PROJECT_ROOT / "data" / "annotations"
    DATASETS_DIR: Path = PROJECT_ROOT / "data" / "datasets"
    
    # Model directories
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    DEPLOYMENT_DIR: Path = PROJECT_ROOT / "models" / "deployment"
    
    # Experiment directories
    EXPERIMENTS_DIR: Path = PROJECT_ROOT / "experiments"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    TEMP_DIR: Path = PROJECT_ROOT / "temp"
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for key, value in self.__dict__.items():
            if isinstance(value, Path):
                value.mkdir(parents=True, exist_ok=True)

# App configuration
@dataclass
class AppConfig:
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    API_WORKERS: int = 1
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

# Main configuration
@dataclass
class FootballHighlightsConfig:
    """Main configuration class combining all configs"""
    VIDEO: VideoConfig = field(default_factory=VideoConfig)
    AUDIO: AudioConfig = field(default_factory=AudioConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    EVENT: EventConfig = field(default_factory=EventConfig)
    PATH: PathConfig = field(default_factory=PathConfig)
    APP: AppConfig = field(default_factory=AppConfig)
    
    # For backward compatibility
    VIDEO_CONFIG: VideoConfig = field(default_factory=VideoConfig)
    
    def __post_init__(self):
        """Initialize backward compatibility aliases"""
        self.VIDEO_CONFIG = self.VIDEO
        
        # Create dynamic attributes for backward compatibility
        object.__setattr__(self, 'EVENT_WEIGHTS', self.EVENT.EVENT_WEIGHTS)
        
        # Also add VIDEO as a simple alias (not property)
        if not hasattr(self, '_video_alias'):
            object.__setattr__(self, '_video_alias', self.VIDEO)

# Create global configuration instance
config = FootballHighlightsConfig()

# Test that directories are created
if __name__ == "__main__":
    print(f"✅ Configuration loaded")
    print(f"📁 Project root: {config.PATH.PROJECT_ROOT}")
    print(f"🎥 Videos directory: {config.PATH.RAW_VIDEO_DIR}")
    print(f"🔊 Audio buffers: {config.AUDIO.KEYWORD_BUFFER_BEFORE}s before, {config.AUDIO.KEYWORD_BUFFER_AFTER}s after")
    print(f"🎬 Clip duration: {config.VIDEO.CLIP_DURATION}s")
    print(f"⚽ Event weights: {list(config.EVENT.EVENT_WEIGHTS.keys())}")
    print(f"⚙️ Event buffers: {config.EVENT.EVENT_BUFFER_BEFORE}s before, {config.EVENT.EVENT_BUFFER_AFTER}s after")
    print(f"🔧 Backward compatibility check:")
    print(f"   config.EVENT_WEIGHTS exists: {hasattr(config, 'EVENT_WEIGHTS')}")
    print(f"   config.EVENT_WEIGHTS type: {type(getattr(config, 'EVENT_WEIGHTS', None))}")

# Export configuration
__all__ = [
    'config',
    'VideoConfig',
    'AudioConfig',
    'ModelConfig',
    'EventConfig',
    'PathConfig',
    'AppConfig',
    'FootballHighlightsConfig'
]