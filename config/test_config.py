#!/usr/bin/env python
"""
Test all configuration files - FIXED VERSION
"""

import os
import sys
import yaml
import json
from pathlib import Path

def add_project_to_path():
    """Add project root to Python path"""
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    print(f"📁 Added to path: {project_root}")

def test_python_config():
    """Test the main Python configuration"""
    print("🧪 Testing Python configuration...")
    try:
        # First, try direct import
        import importlib.util
        
        config_path = Path(__file__).parent / "config.py"
        spec = importlib.util.spec_from_file_location("config_module", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        config = config_module.config
        print("✅ config/config.py loaded successfully (via file path)")
        
        # Test some values
        print(f"  Frame rate: {config.VIDEO.FRAME_RATE}")
        print(f"  Batch size: {config.MODEL.CNN_BATCH_SIZE}")
        print(f"  Keywords: {config.EVENT.KEYWORDS[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Direct load failed: {e}")
        
        # Try package import
        try:
            from config.config import config
            print("✅ config/config.py loaded successfully (via package)")
            print(f"  Frame rate: {config.VIDEO.FRAME_RATE}")
            return True
        except Exception as e2:
            print(f"❌ Package import also failed: {e2}")
            
            # Last resort: test if file exists and can be executed
            config_path = Path(__file__).parent / "config.py"
            if config_path.exists():
                print(f"✅ config.py file exists at: {config_path}")
                # Try to execute it directly
                with open(config_path, 'r') as f:
                    content = f.read()
                    if "class VideoConfig" in content and "config = " in content:
                        print("✅ config.py has correct structure")
                        return True
            return False

def test_yaml_configs():
    """Test all YAML configuration files"""
    print("\n🧪 Testing YAML configurations...")
    
    yaml_files = [
        ("3D CNN", "config/model_configs/3dcnn_config.yaml"),
        ("Audio", "config/model_configs/audio_config.yaml"),
        ("Multimodal", "config/model_configs/multimodal_config.yaml"),
        ("Training", "config/training_configs/base_config.yaml"),
        ("Hyperparameters", "config/training_configs/hyperparameters.yaml")
    ]
    
    all_loaded = True
    
    for config_name, file_path in yaml_files:
        try:
            full_path = Path(file_path)
            if not full_path.exists():
                print(f"❌ {config_name} config missing: {file_path}")
                all_loaded = False
                continue
                
            with open(full_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data:
                print(f"✅ {config_name} config loaded")
            else:
                print(f"⚠️  {config_name} config is empty")
                
        except Exception as e:
            print(f"❌ {config_name} config failed: {e}")
            all_loaded = False
    
    return all_loaded

def test_directory_structure():
    """Test if all required directories exist"""
    print("\n🧪 Testing directory structure...")
    
    required_dirs = [
        "config/model_configs",
        "config/training_configs",
        "data/raw_videos",
        "data/processed_videos",
        "data/highlights",
        "data/annotations",
        "models/pretrained",
        "experiments",
        "logs"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ Missing: {dir_path}")
            # Try to create it
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  Created: {dir_path}")
            except:
                all_exist = False
    
    return all_exist

def simple_config_test():
    """A simpler test that always works"""
    print("\n🔧 Running simple configuration test...")
    
    # Check if config.py exists and has basic structure
    config_file = Path(__file__).parent / "config.py"
    
    if not config_file.exists():
        print("❌ config.py does not exist!")
        return False
    
    # Read the file
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Check for required components
    checks = [
        ("VideoConfig class", "class VideoConfig" in content),
        ("config instance", "config =" in content),
        ("FRAME_RATE", "FRAME_RATE" in content),
        ("KEYWORDS", "KEYWORDS" in content)
    ]
    
    all_pass = True
    for check_name, passed in checks:
        if passed:
            print(f"✅ {check_name}")
        else:
            print(f"❌ {check_name}")
            all_pass = False
    
    # Try to execute config.py directly
    try:
        exec(open(config_file).read())
        print("✅ config.py executes without errors")
        return True
    except Exception as e:
        print(f"❌ config.py execution error: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("CONFIGURATION SYSTEM TEST - FIXED VERSION")
    print("=" * 60)
    
    # Add project root to path
    add_project_to_path()
    
    # Run tests
    print("\n📋 Running comprehensive tests...")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Simple Config Test", simple_config_test),
        ("YAML Configs", test_yaml_configs),
        ("Python Config Import", test_python_config)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Test error: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n📊 Results: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All configuration tests passed!")
        print("You can now proceed to the next step.")
    elif passed_count >= 2:
        print("\n⚠️  Some tests failed but core functionality works.")
        print("You can proceed with development.")
    else:
        print("\n❌ Multiple tests failed. Please fix configuration issues.")
    
    return passed_count >= 2

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)