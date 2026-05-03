#!/usr/bin/env python3
"""
Test the MFCC audio model fix
"""

import torch
import sys
sys.path.insert(0, '.')

from src.feature_extraction.multimodal_fusion import create_multimodal_model

def test_mfcc_model():
    print("🧪 Testing MFCC-compatible LightMultiModal3DCNN...")
    
    # Create model with fixed audio model
    model = create_multimodal_model('light', num_classes=4)
    model.eval()
    
    # Test inputs
    batch_size = 2
    
    # Video: [batch, 3, 16, 90, 160]
    video_input = torch.randn(batch_size, 3, 16, 90, 160)
    
    # Audio: Test different MFCC formats
    print("\n🎯 Test 1: MFCC with channel [batch, 1, 13, 100]")
    audio_input = torch.randn(batch_size, 1, 13, 100)
    
    try:
        with torch.no_grad():
            output, features = model(video_input, audio_input)
            print(f"✅ Success!")
            print(f"📊 Output shape: {output.shape} (expected: [{batch_size}, 4])")
            print(f"📊 Features shape: {features.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Test 2: MFCC without channel [batch, 13, 100]")
    audio_input = torch.randn(batch_size, 13, 100)
    
    try:
        with torch.no_grad():
            output, features = model(video_input, audio_input)
            print(f"✅ Success!")
            print(f"📊 Output shape: {output.shape} (expected: [{batch_size}, 4])")
            print(f"📊 Features shape: {features.shape}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mfcc_model()