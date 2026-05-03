#!/usr/bin/env python
"""
Simple test script - no encoding issues
"""

import sys
from pathlib import Path

def test_config():
    """Test if config can be imported"""
    print("🧪 Testing configuration...")
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"Project root: {project_root}")
    
    # Add to Python path
    sys.path.insert(0, str(project_root))
    
    # Try to import config
    try:
        from config.config import config
        print("✅ SUCCESS: Config imported!")
        print(f"   Frame rate: {config.VIDEO.FRAME_RATE}")
        print(f"   Batch size: {config.MODEL.CNN_BATCH_SIZE}")
        print(f"   Keywords: {len(config.EVENT.KEYWORDS)}")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def create_backup_config():
    """Create a backup config if current one has issues"""
    print("\n🛠️ Creating backup config...")
    
    backup_config = '''
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class VideoConfig:
    FRAME_RATE = 30
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

@dataclass
class AudioConfig:
    SAMPLE_RATE = 16000
    LANGUAGE = "en"

@dataclass
class ModelConfig:
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001

@dataclass
class EventConfig:
    KEYWORDS = ["goal", "red card", "yellow card", "foul", "penalty"]

@dataclass
class PathConfig:
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_VIDEOS = PROJECT_ROOT / "data" / "raw_videos"
    HIGHLIGHTS = PROJECT_ROOT / "data" / "highlights"
    
    def __post_init__(self):
        self.RAW_VIDEOS.mkdir(parents=True, exist_ok=True)
        self.HIGHLIGHTS.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    VIDEO = VideoConfig()
    AUDIO = AudioConfig()
    MODEL = ModelConfig()
    EVENT = EventConfig()
    PATH = PathConfig()

config = Config()
print("✅ Backup config created")
'''
    
    config_file = Path(__file__).parent / "config" / "config.py"
    
    # Backup existing config
    if config_file.exists():
        backup_file = config_file.with_suffix('.py.backup')
        config_file.rename(backup_file)
        print(f"📦 Backed up existing config to: {backup_file}")
    
    # Write new config
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(backup_config)
    
    print(f"📝 Created new config: {config_file}")
    
    # Test it
    try:
        exec(backup_config)
        print("✅ Backup config works!")
        return True
    except Exception as e:
        print(f"❌ Backup config failed: {e}")
        return False

def main():
    print("=" * 60)
    print("SIMPLE CONFIG TEST")
    print("=" * 60)
    
    if test_config():
        print("\n" + "=" * 60)
        print("✅ CONFIGURATION IS WORKING!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️ CONFIG HAS ISSUES - Creating backup...")
        print("=" * 60)
        
        if create_backup_config():
            print("\n✅ Backup config created. Testing again...")
            test_config()

if __name__ == "__main__":
    main()