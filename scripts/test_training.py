# scripts/test_training.py
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def create_dummy_data(num_samples=32, num_classes=4):
    """Create dummy data for testing"""
    # Video: (batch, channels, frames, height, width)
    video_data = torch.randn(num_samples, 3, 16, 90, 160)
    
    # Audio: (batch, channels, samples)
    audio_data = torch.randn(num_samples, 1, 16000)
    
    # Labels
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Create dataset as list of tuples (for compatibility with your trainer)
    dataset = [(video_data[i], audio_data[i], labels[i]) for i in range(num_samples)]
    
    return dataset

def create_dummy_model(num_classes=4):
    """Create a very simple dummy model for testing"""
    class DummyModel(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            self.video_conv = nn.Conv3d(3, 8, kernel_size=3, padding=1)
            self.audio_conv = nn.Conv1d(1, 8, kernel_size=3, padding=1)
            self.fc = nn.Linear(16, num_classes)
            
        def forward(self, video, audio):
            # Simple forward pass
            video_out = self.video_conv(video).mean(dim=(2, 3, 4))
            audio_out = self.audio_conv(audio).mean(dim=2)
            combined = torch.cat([video_out, audio_out], dim=1)
            return self.fc(combined)
    
    return DummyModel(num_classes)

def simple_train_test():
    """Simple training test without complex trainer"""
    print("=" * 60)
    print("Testing basic model training...")
    print("=" * 60)
    
    # Create dummy data
    num_samples = 32
    num_classes = 4
    
    video_data = torch.randn(num_samples, 3, 16, 90, 160)
    audio_data = torch.randn(num_samples, 1, 16000)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    print(f"Created dummy data:")
    print(f"  Video shape: {video_data.shape}")
    print(f"  Audio shape: {audio_data.shape}")
    print(f"  Labels shape: {labels.shape}")
    
    # Create model
    model = create_dummy_model(num_classes)
    print(f"\nModel created:")
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    with torch.no_grad():
        batch_video = video_data[:2]
        batch_audio = audio_data[:2]
        batch_labels = labels[:2]
        
        outputs = model(batch_video, batch_audio)
        print(f"  Output shape: {outputs.shape}")
        print(f"  Expected: (2, {num_classes})")
        
        _, preds = torch.max(outputs, 1)
        print(f"  Predictions: {preds}")
        print(f"  True labels: {batch_labels}")
    
    # Simple training loop
    print("\n" + "=" * 60)
    print("Running simple training loop (2 batches)...")
    print("=" * 60)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    
    for batch_idx in range(2):
        # Get batch
        start_idx = batch_idx * 2
        end_idx = start_idx + 2
        
        batch_video = video_data[start_idx:end_idx]
        batch_audio = audio_data[start_idx:end_idx]
        batch_labels = labels[start_idx:end_idx]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(batch_video, batch_audio)
        loss = criterion(outputs, batch_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, preds = torch.max(outputs, 1)
        accuracy = (preds == batch_labels).float().mean().item()
        
        print(f"  Batch {batch_idx + 1}: loss={loss.item():.4f}, accuracy={accuracy:.2%}")
    
    print("\n" + "=" * 60)
    print("✅ Basic training test PASSED!")
    print("=" * 60)
    
    return True

def test_with_actual_trainer():
    """Test with your actual trainer"""
    print("\n" + "=" * 60)
    print("Testing with actual trainer...")
    print("=" * 60)
    
    try:
        # Try to import your trainer
        from src.model_training.trainer import ModelTrainer, create_trainer
        
        # Create dummy dataset
        dataset = create_dummy_data(64, 4)
        
        # Create data loaders
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        test_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        # Create model
        model = create_dummy_model(4)
        
        # Test with trainer
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            experiment_name="test_training",
            use_wandb=False,
            device='cpu'
        )
        
        print("✅ Trainer initialized successfully!")
        
        # Test one epoch
        print("\nTesting one training epoch...")
        train_metrics = trainer.train_epoch()
        print(f"  Training metrics: {train_metrics}")
        
        print("\nTesting validation...")
        val_metrics = trainer.validate()
        print(f"  Validation metrics: {val_metrics}")
        
        print("\n" + "=" * 60)
        print("✅ Trainer test PASSED!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"❌ Trainer test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 Starting training tests...")
    
    # Test 1: Basic model and training
    if not simple_train_test():
        print("❌ Basic test failed!")
        return
    
    # Test 2: With actual trainer (optional)
    print("\n\nProceeding to trainer test...")
    try:
        test_with_actual_trainer()
    except:
        print("⚠️  Could not test with trainer, but basic training works!")
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Update your config with smaller dimensions (160x90 instead of 320x180)")
    print("2. Reduce batch size to 2")
    print("3. Run: python scripts/train_model.py --verbose --num_workers 0 --batch_size 2 --num_epochs 5")

if __name__ == "__main__":
    main()