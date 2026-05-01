#!/usr/bin/env python3
"""
Test script to debug model dimensions
"""

import torch
from src.feature_extraction.multimodal_fusion import create_multimodal_model
from config.config import config

def test_model_dimensions():
    print("🧪 Testing model dimensions...")
    
    # Create model
    model = create_multimodal_model('light', num_classes=4)
    model.eval()
    
    # Create test input with EXACTLY the expected dimensions
    batch_size = 2
    
    # Video input: [batch, 3, 16, 90, 160]
    print(f"🎥 Creating video input: [{batch_size}, 3, 16, 90, 160]")
    video_input = torch.randn(batch_size, 3, 16, 90, 160)
    
    # Audio input: [batch, 1, 13, 100] - MFCC features
    print(f"🔊 Creating audio input: [{batch_size}, 1, 13, 100]")
    audio_input = torch.randn(batch_size, 1, 13, 100)
    
    print("\n🧪 Running model inference...")
    try:
        with torch.no_grad():
            output, features = model(video_input, audio_input)
            print(f"✅ Model inference successful!")
            print(f"📊 Output shape: {output.shape}")
            print(f"📊 Features shape: {features.shape}")
            print(f"🎯 Expected output shape: [{batch_size}, 4]")
            
            if output.shape == torch.Size([batch_size, 4]):
                print("✅ Output dimensions are CORRECT!")
            else:
                print(f"❌ Output dimensions are WRONG! Got {output.shape}, expected [{batch_size}, 4]")
                
    except Exception as e:
        print(f"❌ Model inference failed: {e}")
        import traceback
        traceback.print_exc()

def test_real_pipeline():
    print("\n" + "="*60)
    print("🧪 Testing real pipeline...")
    
    from src.video_processing.preprocessor import VideoProcessor, VideoClip
    import numpy as np
    
    processor = VideoProcessor(config)
    
    # Create a dummy clip with correct dimensions
    frames = np.random.randint(0, 256, (16, 90, 160, 3), dtype=np.uint8)
    clip = VideoClip(
        frames=frames,
        start_time=0.0,
        end_time=1.0,
        fps=25.0,
        frame_count=16
    )
    
    print(f"🎥 Created clip: {clip.frame_count} frames, {clip.frames.shape}")
    
    # Test tensor conversion
    from src.video_processing.preprocessor import frames_to_tensor
    tensor = frames_to_tensor(clip.frames)
    print(f"📊 Tensor shape: {tensor.shape}")
    print(f"✅ Expected: [3, 16, 90, 160]")
    
    if tensor.shape == torch.Size([3, 16, 90, 160]):
        print("✅ Tensor dimensions are CORRECT!")
    else:
        print(f"❌ Tensor dimensions are WRONG! Got {tensor.shape}")

if __name__ == "__main__":
    test_model_dimensions()
    test_real_pipeline()