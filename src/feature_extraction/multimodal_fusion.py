# src/feature_extraction/multimodal_fusion.py - FINAL CORRECTED VERSION
"""
Multimodal fusion models for football event detection - LIGHTWEIGHT VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

from src.feature_extraction.cnn3d_model import Sports3DCNN
from src.feature_extraction.audio_model import AudioCNN
from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class LightAttentionFusion(nn.Module):
    """Lightweight attention-based fusion of multimodal features"""
    
    def __init__(self, visual_dim: int, audio_dim: int, hidden_dim: int = 128):
        super(LightAttentionFusion, self).__init__()
        
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        
        # Lightweight attention mechanisms
        self.visual_attention = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.audio_attention = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Lightweight fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + audio_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # Reduced from 0.5
        )
        
        self.fusion_dim = hidden_dim
    
    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multimodal input
        
        Args:
            visual_input: Visual input tensor [batch, 3, 16, 90, 160]
            audio_input: Audio input tensor [batch, 1, 13, 100] or [batch, 13, 100] (MFCC)
        """
        print(f"🔍 MULTIMODAL MODEL: Visual input shape: {visual_input.shape}")
        print(f"🔍 MULTIMODAL MODEL: Audio input shape: {audio_input.shape}")
    
        # Extract features from both modalities
        visual_output, visual_features = self.visual_model(visual_input)
        audio_output, audio_features = self.audio_model(audio_input)
        
        print(f"🔍 MULTIMODAL MODEL: Visual features shape: {visual_features.shape}")
        print(f"🔍 MULTIMODAL MODEL: Audio features shape: {audio_features.shape}")
        
        # Reduce feature dimensions if needed (safety check)
        if visual_features.size(1) > 128:
            print(f"⚠️ Trimming visual features from {visual_features.shape[1]} to 128")
            visual_features = visual_features[:, :128]
        
        if audio_features.size(1) > 64:
            print(f"⚠️ Trimming audio features from {audio_features.shape[1]} to 64")
            audio_features = audio_features[:, :64]
        elif audio_features.size(1) < 64:
            print(f"⚠️ Padding audio features from {audio_features.shape[1]} to 64")
            padding = torch.zeros(audio_features.size(0), 64 - audio_features.size(1), 
                                device=audio_features.device)
            audio_features = torch.cat([audio_features, padding], dim=1)
        
        # Fuse features
        if self.fusion_method == 'attention':
            fused_features = self.fusion(visual_features, audio_features)
        else:  # concat
            fused_features = torch.cat([visual_features, audio_features], dim=1)
        
        print(f"🔍 MULTIMODAL MODEL: Fused features shape: {fused_features.shape}")
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, fused_features


class LightMultiModal3DCNN(nn.Module):
    """LIGHTWEIGHT Multimodal 3D CNN with audio-visual fusion"""
    
    def __init__(
        self,
        visual_model: nn.Module,
        audio_model: nn.Module,
        num_classes: int = 4,
        fusion_method: str = 'attention',
        dropout_rate: float = 0.3
    ):
        super(LightMultiModal3DCNN, self).__init__()
        
        self.visual_model = visual_model
        self.audio_model = audio_model
        self.fusion_method = fusion_method
        self.num_classes = num_classes
    
        # UPDATED: Get feature dimensions - AUDIO DIM IS NOW 64
        visual_dim = 128
        audio_dim = 64  # From the fixed audio model
        
        # Fusion module
        if fusion_method == 'attention':
            self.fusion = LightAttentionFusion(visual_dim, audio_dim, hidden_dim=128)
            fusion_dim = self.fusion.fusion_dim
        elif fusion_method == 'concat':
            fusion_dim = visual_dim + audio_dim  # 128 + 64 = 192
            self.fusion = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # UPDATED: Classifier input dimension should match fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        # Log dimensions
        logger.info(f"Initialized LightMultiModal3DCNN:")
        logger.info(f"  Visual features: {visual_dim} dimensions")
        logger.info(f"  Audio features: {audio_dim} dimensions (MFCC)")
        logger.info(f"  Fusion method: {fusion_method}")
        logger.info(f"  Classifier input: {fusion_dim} dimensions")
    
    def _initialize_weights(self):
        """Initialize weights for fusion and classifier layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multimodal input
        
        Args:
            visual_input: Visual input tensor (batch, channels, depth, height, width)
            audio_input: Audio input tensor (batch, channels, features, time)
            
        Returns:
            output: Classification logits
            fused_features: Fused features before classification
        """
        # Extract features from both modalities
        visual_output, visual_features = self.visual_model(visual_input)
        audio_output, audio_features = self.audio_model(audio_input)
        
        # Reduce feature dimensions if needed
        if visual_features.size(1) > 128:
            visual_features = visual_features[:, :128]
        if audio_features.size(1) > 64:
            audio_features = audio_features[:, :64]
        
        # Fuse features
        if self.fusion_method == 'attention':
            fused_features = self.fusion(visual_features, audio_features)
        else:  # concat
            fused_features = torch.cat([visual_features, audio_features], dim=1)
        
        # Classification
        output = self.classifier(fused_features)
        
        return output, fused_features
    
    def forward_multimodal(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Forward pass returning only logits (for training compatibility)"""
        output, _ = self.forward(visual_input, audio_input)
        return output


class SimpleMultimodalModel(nn.Module):
    """Simple multimodal model for MFCC audio features (13x100)"""
    
    def __init__(self, num_classes: int = 4):
        super(SimpleMultimodalModel, self).__init__()
        
        # Simple visual branch for 90x160 input
        self.visual_conv = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),  # Pool spatial: 90x160 -> 45x80
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d((2, 2, 2)),  # Pool temporal+spatial: 16->8, 45x80->22x40
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # Output: [batch, 32, 1, 1, 1]
        )
        
        # Audio branch for MFCC features: [batch, 13, 100] or [batch, 1, 13, 100]
        # Using Conv2d for 2D MFCC features
        self.audio_conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),  # Input: [batch, 1, 13, 100]
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 13x100 -> 7x50
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 7x50 -> 4x25
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Output: [batch, 32, 1, 1]  # FIXED: was AdaptiveAvgPool1d
        )
        
        # Alternative Conv1d path if input is [batch, 13, 100] without channel dim
        self.audio_conv1d = nn.Sequential(
            nn.Conv1d(13, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 100 -> 50
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Output: [batch, 32, 1]
        )
        
        # Fusion and classification
        # Visual: 32 features, Audio: 32 features = 64 total
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )
        
        # Initialize
        self._initialize_weights()
        
        params = sum(p.numel() for p in self.parameters())
        logger.info(f"Created SimpleMultimodalModel (MFCC) with {params:,} parameters")
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with flexible audio input handling"""
        # Visual features: input shape (batch, 3, 16, 90, 160)
        visual_features = self.visual_conv(visual_input)
        visual_features = visual_features.view(visual_features.size(0), -1)  # [batch, 32]
        
        # Audio features: handle different input formats
        batch_size = audio_input.size(0)
        audio_dim = audio_input.dim()
        
        if audio_dim == 4:  # [batch, 1, 13, 100] - has channel dimension
            # Use Conv2d path
            audio_features = self.audio_conv(audio_input)
            audio_features = audio_features.view(batch_size, -1)  # [batch, 32]
            
        elif audio_dim == 3:  # [batch, 13, 100] - no channel dimension
            # Check if it's MFCC (13 coefficients)
            if audio_input.size(1) == 13:
                # Option 1: Add channel dim and use Conv2d
                audio_input_4d = audio_input.unsqueeze(1)  # [batch, 1, 13, 100]
                audio_features = self.audio_conv(audio_input_4d)
                audio_features = audio_features.view(batch_size, -1)
            else:
                # Option 2: Use Conv1d (for raw audio or other 1D features)
                audio_features = self.audio_conv1d(audio_input)
                audio_features = audio_features.view(batch_size, -1)
                
        elif audio_dim == 2:  # [batch, 16000] - raw waveform
            # Reshape to [batch, 1, 16000] and would need different processing
            # For now, create dummy features
            audio_features = torch.zeros(batch_size, 32, device=audio_input.device)
            logger.warning("Raw audio waveform detected - using dummy audio features")
        else:
            raise ValueError(f"Unsupported audio input dimension: {audio_dim}, shape: {audio_input.shape}")
        
        # Concatenate visual and audio features
        combined = torch.cat([visual_features, audio_features], dim=1)
        
        # Classify
        output = self.classifier(combined)
        
        return output
    
    def forward_multimodal(self, visual_input: torch.Tensor, audio_input: torch.Tensor) -> torch.Tensor:
        """Alias for forward"""
        return self.forward(visual_input, audio_input)


def create_light_sports_model(num_classes=4):
    """Create a lightweight Sports3DCNN with FIXED dimensions"""
    class LightSports3DCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # Smaller 3D CNN - FIXED FOR 90x160 INPUT
            self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm3d(16)
            self.pool1 = nn.MaxPool3d((1, 2, 2))  # Pool spatial: 90x160 -> 45x80
            
            self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm3d(32)
            self.pool2 = nn.MaxPool3d((2, 2, 2))  # Pool temporal+spatial: 16->8, 45x80->22x40
            
            self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm3d(64)
            self.pool3 = nn.MaxPool3d((2, 2, 2))  # 8->4, 22x40->11x20
            
            # CRITICAL FIX: Calculate the correct flattened size
            # After 3 pooling layers:
            # Temporal: 16 -> 8 -> 4 (frames)
            # Height: 90 -> 45 -> 22 -> 11
            # Width: 160 -> 80 -> 40 -> 20
            # Channels: 64
            
            # 64 channels * 4 frames * 11 height * 20 width = 56,320
            self.expected_flattened_size = 64 * 4 * 11 * 20
            self.fc1 = nn.Linear(self.expected_flattened_size, 128)
            self.fc2 = nn.Linear(128, num_classes)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            # DEBUG: Log input shape
            print(f"🔍 Model input shape: {x.shape}")
            
            # Return both output and features
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            
            # DEBUG: Log shape before flattening
            print(f"🔍 Shape before flattening: {x.shape}")
            
            # Flatten with validation
            features = x.view(x.size(0), -1)
            
            # DEBUG: Log flattened size
            print(f"🔍 Flattened size: {features.shape[1]}")
            print(f"🔍 Expected size: {self.expected_flattened_size}")
            
            # FIX: If features are too large, trim them
            if features.shape[1] > self.expected_flattened_size:
                print(f"⚠️ Trimming features from {features.shape[1]} to {self.expected_flattened_size}")
                features = features[:, :self.expected_flattened_size]
            elif features.shape[1] < self.expected_flattened_size:
                print(f"⚠️ Padding features from {features.shape[1]} to {self.expected_flattened_size}")
                padding = torch.zeros(x.size(0), self.expected_flattened_size - features.shape[1], 
                                     device=features.device)
                features = torch.cat([features, padding], dim=1)
            
            features = self.dropout(features)
            
            # Classification
            x = self.relu(self.fc1(features))
            x = self.dropout(x)
            output = self.fc2(x)
            
            return output, features
    
    return LightSports3DCNN(num_classes)


def create_light_audio_model(num_classes=4):
    """Create a lightweight AudioCNN for MFCC features (13x100)"""
    class LightAudioCNN(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            
            # FIXED: Process MFCC features (13x100) using Conv2d
            # Input: [batch, 1, 13, 100] for MFCC with channel
            # Input: [batch, 13, 100] for MFCC without channel
            
            # Conv layers for MFCC features
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # [batch, 16, 13, 100]
            self.bn1 = nn.BatchNorm2d(16)
            self.pool1 = nn.MaxPool2d(2)  # [batch, 16, 6, 50]
            
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # [batch, 32, 6, 50]
            self.bn2 = nn.BatchNorm2d(32)
            self.pool2 = nn.MaxPool2d(2)  # [batch, 32, 3, 25]
            
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # [batch, 64, 3, 25]
            self.bn3 = nn.BatchNorm2d(64)
            self.pool3 = nn.AdaptiveAvgPool2d(1)  # [batch, 64, 1, 1]
            
            # Calculate flattened size: 64 * 1 * 1 = 64
            self.fc1 = nn.Linear(64, 64)
            self.fc2 = nn.Linear(64, num_classes)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.3)
            
        def forward(self, x):
            """
            Forward pass for MFCC audio features
            
            Args:
                x: Input tensor of shape:
                   - [batch, 1, 13, 100] (MFCC with channel)
                   - [batch, 13, 100] (MFCC without channel)
            """
            # DEBUG: Log input shape
            print(f"🔍 AUDIO MODEL: Input shape: {x.shape}")
            
            # Ensure we have channel dimension
            if x.dim() == 3:  # [batch, 13, 100] - no channel dim
                print(f"🔍 AUDIO MODEL: Adding channel dimension")
                x = x.unsqueeze(1)  # [batch, 1, 13, 100]
            
            # Process through conv layers
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool2(x)
            
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool3(x)
            
            # Flatten: [batch, 64, 1, 1] -> [batch, 64]
            features = x.view(x.size(0), -1)
            print(f"🔍 AUDIO MODEL: Features shape after conv: {features.shape}")
            
            features = self.dropout(features)
            
            # Classification
            x = self.relu(self.fc1(features))
            x = self.dropout(x)
            output = self.fc2(x)
            
            return output, features
    
    return LightAudioCNN(num_classes)


def create_multimodal_model(
    model_type: str = 'simple',  # Changed default to 'simple' for lightweight
    num_classes: int = 4,  # Changed default to 4
    pretrained: bool = False,  # Changed default to False for lightweight
    fusion_method: str = 'concat',  # Changed default to simpler fusion
    **kwargs
) -> nn.Module:
    """
    Factory function to create multimodal models - UPDATED FOR LIGHTWEIGHT
    
    Args:
        model_type: Type of model to create ('simple' recommended)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        fusion_method: Fusion method for multimodal models
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized model
    """
    model_registry = {
        'multimodal': LightMultiModal3DCNN,  # Updated to light version
        'simple': SimpleMultimodalModel,
        'light': LightMultiModal3DCNN,  # Alias for light
    }
    
    if model_type not in model_registry:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(model_registry.keys())}")
    
    if model_type in ['multimodal', 'light']:
        # Create lightweight visual and audio models
        visual_model = create_light_sports_model(num_classes)
        audio_model = create_light_audio_model(num_classes)
        
        # Create multimodal model
        model = LightMultiModal3DCNN(
            visual_model=visual_model,
            audio_model=audio_model,
            num_classes=num_classes,
            fusion_method=fusion_method,
            dropout_rate=kwargs.get('dropout_rate', 0.3)
        )
        
    elif model_type == 'simple':
        model = SimpleMultimodalModel(num_classes=num_classes)
    
    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Created {model_type} model with {num_classes} classes")
    logger.info(f"Total parameters: {params:,}")
    
    if params > 1_000_000:
        logger.warning(f"Model has {params:,} parameters. Consider using 'simple' for CPU training.")
    
    return model


def load_model(model_path: str, device: str = None) -> nn.Module:
    """
    Load trained model from file
    
    Args:
        model_path: Path to model file
        device: Device to load model to
        
    Returns:
        Loaded model
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model configuration
        model_config = checkpoint.get('config', {})
        model_type = model_config.get('MODEL', {}).get('FUSION_METHOD', 'simple')  # Default to simple
        num_classes = model_config.get('MODEL', {}).get('num_classes', 4)
        
        # Create model
        model = create_multimodal_model(
            model_type=model_type,
            num_classes=num_classes,
            pretrained=False
        )
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {model_path}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def test_lightweight_models():
    """Test lightweight model functions with MFCC inputs"""
    print("Testing lightweight multimodal models with MFCC...")
    
    # Test with your actual input dimensions
    batch_size = 2
    
    # Video: [batch, 3, 16, 90, 160]
    video_input = torch.randn(batch_size, 3, 16, 90, 160)
    
    # Test different audio formats
    print("\nTesting SimpleMultimodalModel with different audio inputs:")
    
    # Test 1: MFCC with channel [batch, 1, 13, 100]
    audio_mfcc_channel = torch.randn(batch_size, 1, 13, 100)
    print(f"\n1. MFCC with channel dim [batch, 1, 13, 100]:")
    model1 = create_multimodal_model('simple', num_classes=4)
    try:
        output1 = model1(video_input, audio_mfcc_channel)
        print(f"   ✅ Output shape: {output1.shape}")
        print(f"   ✅ Model parameters: {sum(p.numel() for p in model1.parameters()):,}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: MFCC without channel [batch, 13, 100]
    audio_mfcc_no_channel = torch.randn(batch_size, 13, 100)
    print(f"\n2. MFCC without channel dim [batch, 13, 100]:")
    model2 = create_multimodal_model('simple', num_classes=4)
    try:
        output2 = model2(video_input, audio_mfcc_no_channel)
        print(f"   ✅ Output shape: {output2.shape}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: Raw audio [batch, 1, 16000]
    audio_raw = torch.randn(batch_size, 1, 16000)
    print(f"\n3. Raw audio [batch, 1, 16000]:")
    model3 = create_multimodal_model('simple', num_classes=4)
    try:
        output3 = model3(video_input, audio_raw)
        print(f"   ✅ Output shape: {output3.shape}")
    except Exception as e:
        print(f"   ⚠️  Raw audio may not work (expected): {e}")
    
    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)
    print("\nRecommended: Use 'simple' model type for CPU training")

# Backward compatibility alias
MultiModal3DCNN = LightMultiModal3DCNN

if __name__ == "__main__":
    test_lightweight_models()