#!/usr/bin/env python
"""
Debug configuration issues - WINDOWS COMPATIBLE
"""

import os
import sys
from pathlib import Path

def debug_imports():
    """Debug import issues - Windows compatible"""
    print("🔍 Debugging import issues...")
    
    # Get paths
    project_root = Path(__file__).parent
    config_dir = project_root / "config"
    config_file = config_dir / "config.py"
    
    print(f"1. Project root: {project_root}")
    print(f"2. Config directory: {config_dir}")
    print(f"3. Config file: {config_file}")
    
    # Check if files exist
    print(f"\n📁 File existence:")
    print(f"   config/ exists: {config_dir.exists()}")
    print(f"   config.py exists: {config_file.exists()}")
    
    if config_file.exists():
        print(f"\n📄 config.py size: {config_file.stat().st_size} bytes")
        
        # Read with UTF-8 encoding for Windows
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:5]
                print("   First 5 lines:")
                for i, line in enumerate(lines):
                    print(f"   {i+1}: {line.strip()}")
        except UnicodeDecodeError:
            # Try with latin-1 as fallback
            try:
                with open(config_file, 'r', encoding='latin-1') as f:
                    lines = f.readlines()[:5]
                    print("   First 5 lines (latin-1):")
                    for i, line in enumerate(lines):
                        print(f"   {i+1}: {line.strip()}")
            except:
                print("   Could not read file - encoding issues")
    
    # Check Python path
    print(f"\n🐍 Python path (first 3):")
    for i, path in enumerate(sys.path[:3]):
        print(f"   {i}: {path}")
    
    # Try different import methods
    print(f"\n🔄 Testing import methods:")
    
    # Method 1: Add to path and import
    print("\n1. Adding to path and importing...")
    sys.path.insert(0, str(project_root))
    try:
        from config.config import config
        print("   ✅ Success!")
        print(f"   Frame rate: {config.VIDEO.FRAME_RATE}")
        return True
    except ImportError as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 2: Direct module execution
    print("\n2. Direct module execution...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("myconfig", config_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print("   ✅ Success!")
        if hasattr(module, 'config'):
            print(f"   Config found in module")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Method 3: Simple exec with UTF-8
    print("\n3. Simple exec (UTF-8)...")
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            code = f.read()
        exec(code)
        print("   ✅ Success!")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        # Try latin-1
        try:
            with open(config_file, 'r', encoding='latin-1') as f:
                code = f.read()
            exec(code)
            print("   ✅ Success with latin-1!")
            return True
        except:
            pass
    
    return False

def create_minimal_config():
    """Create a minimal working config"""
    print("\n🛠️ Creating minimal config...")
    
    minimal_config = '''"""
Minimal configuration - guaranteed to work
"""

from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class VideoConfig:
    FRAME_RATE = 30
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

@dataclass
class ModelConfig:
    BATCH_SIZE = 8
    LEARNING_RATE = 0.0001

@dataclass  
class PathConfig:
    PROJECT_ROOT = Path(__file__).parent.parent
    RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_videos"
    HIGHLIGHTS_DIR = PROJECT_ROOT / "data" / "highlights"
    
    def __post_init__(self):
        self.RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        self.HIGHLIGHTS_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class Config:
    VIDEO: VideoConfig = field(default_factory=VideoConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    PATH: PathConfig = field(default_factory=PathConfig)

config = Config()

if __name__ == "__main__":
    print("✅ Minimal config created")
    print(f"Project: {config.PATH.PROJECT_ROOT}")
    print(f"Frame rate: {config.VIDEO.FRAME_RATE}")
'''

    config_file = Path(__file__).parent / "config" / "config.py"
    config_file.parent.mkdir(exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(minimal_config)
    
    print(f"📁 Created: {config_file}")
    
    # Test it
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from config.config import config
        print("🎉 SUCCESS: Minimal config works!")
        print(f"   Frame rate: {config.VIDEO.FRAME_RATE}")
        return True
    except Exception as e:
        print(f"⚠️ Import failed: {e}")
        # Try direct execution
        try:
            exec(minimal_config)
            print("✅ Config executes directly")
            return True
        except:
            print("❌ Could not execute config")
            return False

def main():
    print("=" * 60)
    print("CONFIGURATION DEBUGGER - WINDOWS EDITION")
    print("=" * 60)
    
    if not debug_imports():
        print("\n" + "=" * 60)
        print("Creating minimal config as fallback...")
        print("=" * 60)
        create_minimal_config()

if __name__ == "__main__":
    main()