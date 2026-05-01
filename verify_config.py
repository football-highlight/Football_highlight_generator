#!/usr/bin/env python
"""
Simple script to verify configuration is working
"""

import os
import sys
from pathlib import Path

def main():
    print("🔍 Verifying configuration setup...")
    print("-" * 40)
    
    # Get project root
    project_root = Path(__file__).parent
    print(f"📁 Project root: {project_root}")
    
    # Check config directory
    config_dir = project_root / "config"
    if config_dir.exists():
        print("✅ Config directory exists")
        
        # Check for config.py
        config_file = config_dir / "config.py"
        if config_file.exists():
            print("✅ config.py exists")
            
            # Try to execute it
            try:
                # Add to path
                sys.path.insert(0, str(project_root))
                
                # Import config
                from config.config import config
                print("🎉 SUCCESS: Configuration imported!")
                print(f"   Frame rate: {config.VIDEO.FRAME_RATE}")
                print(f"   Batch size: {config.MODEL.CNN_BATCH_SIZE}")
                print(f"   Keywords: {len(config.EVENT.KEYWORDS)}")
                
                # Check directories
                print("\n📁 Checking directories:")
                dirs_to_check = [
                    config.PATH.RAW_VIDEO_DIR,
                    config.PATH.HIGHLIGHTS_DIR,
                    config.PATH.EXPERIMENTS_DIR
                ]
                
                for dir_path in dirs_to_check:
                    if dir_path.exists():
                        print(f"   ✅ {dir_path.name}")
                    else:
                        print(f"   ❌ {dir_path.name} (missing)")
                        
                return True
                
            except ImportError as e:
                print(f"❌ Import error: {e}")
                print("\n🔧 Trying alternative import method...")
                
                # Try direct execution
                try:
                    exec(open(config_file).read())
                    print("✅ config.py executes directly")
                    return True
                except Exception as e2:
                    print(f"❌ Execution error: {e2}")
                    return False
        else:
            print("❌ config.py missing")
            return False
    else:
        print("❌ Config directory missing")
        return False

if __name__ == "__main__":
    if main():
        print("\n" + "=" * 40)
        print("✅ Configuration setup is WORKING!")
        print("=" * 40)
        sys.exit(0)
    else:
        print("\n" + "=" * 40)
        print("❌ Configuration has ISSUES")
        print("=" * 40)
        sys.exit(1)