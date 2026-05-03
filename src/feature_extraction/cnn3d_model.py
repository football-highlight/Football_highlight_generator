"""
3D CNN models for football video analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights
from typing import Tuple, Optional, List, Dict
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpatialAttention3D(nn.Module):
    """3D Spatial Attention Module"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super(SpatialAttention3D, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attention = avg_out + max_out
        return x * attention


class TemporalAttention3D(nn.Module):
    """3D Temporal Attention Module"""
    
    def __init__(self, in_channels: int, num_frames: int = 16):
        super(TemporalAttention3D, self).__init__()
        
        self.num_frames = num_frames
        self.conv1 = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels // 8, in_channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))  # Pool spatial dimensions
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, T, H, W = x.shape
        
        # Pool spatial dimensions
        spatial_pooled = self.pool(x)  # (B, C, T, 1, 1)
        spatial_pooled = spatial_pooled.view(batch_size, channels, T)
        
        # Apply attention
        attention = F.softmax(spatial_pooled, dim=2)
        attention = attention.view(batch_size, channels, T, 1, 1)
        
        return x * attention


class Sports3DCNN(nn.Module):
    """3D CNN for sports action recognition with attention"""
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        num_frames: int = 16,
        dropout_rate: float = 0.5,
        use_attention: bool = True
    ):
        super(Sports3DCNN, self).__init__()
        
        self.num_frames = num_frames
        self.use_attention = use_attention
        
        # Load pretrained R3D-18
        if pretrained:
            logger.info("Loading pretrained R3D-18 weights")
            weights = R3D_18_Weights.DEFAULT
            self.backbone = r3d_18(weights=weights)
            
            # Freeze early layers if needed
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            self.backbone = r3d_18(weights=None)
        
        # Get the number of features in the final layer
        num_features = self.backbone.fc.in_features
        
        # Remove the original fc layer
        self.backbone.fc = nn.Identity()
        
        # Add attention modules
        if use_attention:
            self.spatial_attention = SpatialAttention3D(512)  # Layer4 output channels
            self.temporal_attention = TemporalAttention3D(512, num_frames)
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized Sports3DCNN with {num_classes} classes")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.parameters()):,}")
        
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            
        Returns:
            output: Classification logits
            features: Extracted features before classification
        """
        batch_size = x.size(0)
        
        # Extract features through backbone
        features = self.backbone.stem(x)
        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.spatial_attention(features)
            features = self.temporal_attention(features)
        
        # Global average pooling
        features = F.adaptive_avg_pool3d(features, (1, 1, 1))
        features = features.view(batch_size, -1)
        
        # Classification
        output = self.classifier(features)
        
        return output, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        _, features = self.forward(x)
        return features


class MultiScale3DCNN(nn.Module):
    """Multi-scale 3D CNN for capturing different temporal resolutions"""
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        scales: List[int] = [8, 16, 32]
    ):
        super(MultiScale3DCNN, self).__init__()
        
        self.scales = scales
        self.models = nn.ModuleList()
        
        # Create a model for each scale
        for scale in scales:
            model = Sports3DCNN(
                num_classes=num_classes,
                pretrained=pretrained,
                num_frames=scale,
                use_attention=True
            )
            self.models.append(model)
        
        # Fusion layer
        total_features = 512 * len(scales)  # 512 features per model
        self.fusion = nn.Sequential(
            nn.Linear(total_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with multiple temporal scales
        
        Args:
            x: Input tensor (batch_size, channels, max_depth, height, width)
            
        Returns:
            output: Classification logits
            features: Concatenated features from all scales
        """
        batch_size, channels, max_depth, H, W = x.shape
        all_features = []
        
        for i, scale in enumerate(self.scales):
            if scale <= max_depth:
                # Extract temporal segment
                start_idx = (max_depth - scale) // 2
                x_scale = x[:, :, start_idx:start_idx + scale, :, :]
                
                # Forward through model
                _, features = self.models[i](x_scale)
                all_features.append(features)
            else:
                # Use the full sequence with interpolation
                x_scale = F.interpolate(
                    x, size=(scale, H, W), mode='trilinear', align_corners=False
                )
                _, features = self.models[i](x_scale)
                all_features.append(features)
        
        # Concatenate features
        combined_features = torch.cat(all_features, dim=1)
        
        # Fusion
        output = self.fusion(combined_features)
        
        return output, combined_features


class OpticalFlowCNN(nn.Module):
    """CNN for optical flow feature extraction"""
    
    def __init__(self, num_classes: int = 10):
        super(OpticalFlowCNN, self).__init__()
        
        # Optical flow has 2 channels (x, y flow)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Block 2
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            
            # Block 3
            nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        
        # Calculate output size
        # Assuming input: (batch, 2, 15, 224, 224)
        # After conv layers: (batch, 128, 15, 28, 28)
        
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for optical flow"""
        # x shape: (batch, 2, depth, height, width)
        
        features = self.conv_layers(x)
        pooled = self.global_pool(features)
        flattened = pooled.view(pooled.size(0), -1)
        
        output = self.fc(flattened)
        
        return output, flattened


class TwoStream3DCNN(nn.Module):
    """Two-stream 3D CNN combining RGB and optical flow"""
    
    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        fusion_method: str = 'concat'  # 'concat', 'sum', 'attention'
    ):
        super(TwoStream3DCNN, self).__init__()
        
        self.fusion_method = fusion_method
        
        # RGB stream
        self.rgb_stream = Sports3DCNN(
            num_classes=num_classes,
            pretrained=pretrained,
            use_attention=True
        )
        
        # Optical flow stream
        self.flow_stream = OpticalFlowCNN(num_classes=num_classes)
        
        # Fusion
        if fusion_method == 'concat':
            self.fusion = nn.Sequential(
                nn.Linear(512 + 128, 256),  # 512 from RGB, 128 from flow
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
        elif fusion_method == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(512 + 128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2),
                nn.Softmax(dim=1)
            )
            self.fusion = nn.Linear(512 + 128, num_classes)
        
    def forward(self, rgb_input: torch.Tensor, flow_input: torch.Tensor) -> torch.Tensor:
        """Forward pass with two streams"""
        
        # Extract features from both streams
        rgb_output, rgb_features = self.rgb_stream(rgb_input)
        flow_output, flow_features = self.flow_stream(flow_input)
        
        # Concatenate features
        combined_features = torch.cat([rgb_features, flow_features], dim=1)
        
        # Apply fusion
        if self.fusion_method == 'concat':
            output = self.fusion(combined_features)
        elif self.fusion_method == 'attention':
            # Compute attention weights
            attention_weights = self.attention(combined_features)
            
            # Apply attention
            rgb_weight = attention_weights[:, 0].unsqueeze(1)
            flow_weight = attention_weights[:, 1].unsqueeze(1)
            
            weighted_rgb = rgb_features * rgb_weight
            weighted_flow = flow_features * flow_weight
            
            # Concatenate weighted features
            weighted_features = torch.cat([weighted_rgb, weighted_flow], dim=1)
            output = self.fusion(weighted_features)
        
        return output


def create_3dcnn_model(
    model_type: str = 'sports_3dcnn',
    num_classes: int = 10,
    pretrained: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create 3D CNN models
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized model
    """
    model_registry = {
        'sports_3dcnn': Sports3DCNN,
        'multi_scale': MultiScale3DCNN,
        'optical_flow': OpticalFlowCNN,
        'two_stream': TwoStream3DCNN,
    }
    
    if model_type not in model_registry:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(model_registry.keys())}")
    
    model_class = model_registry[model_type]
    model = model_class(num_classes=num_classes, pretrained=pretrained, **kwargs)
    
    logger.info(f"Created {model_type} model with {num_classes} classes")
    
    return model


def test_models():
    """Test function for 3D CNN models"""
    print("Testing 3D CNN models...")
    
    # Test Sports3DCNN
    print("\n1. Testing Sports3DCNN...")
    model = Sports3DCNN(num_classes=10, pretrained=False)
    
    # Create dummy input
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    output, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print("✅ Sports3DCNN test passed")
    
    # Test MultiScale3DCNN
    print("\n2. Testing MultiScale3DCNN...")
    model = MultiScale3DCNN(num_classes=10, pretrained=False, scales=[8, 16])
    
    dummy_input = torch.randn(2, 3, 16, 224, 224)
    output, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print("✅ MultiScale3DCNN test passed")
    
    # Test OpticalFlowCNN
    print("\n3. Testing OpticalFlowCNN...")
    model = OpticalFlowCNN(num_classes=10)
    
    dummy_input = torch.randn(2, 2, 15, 224, 224)  # 2 channels for optical flow
    output, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print("✅ OpticalFlowCNN test passed")
    
    print("\n✅ All 3D CNN model tests passed!")


if __name__ == "__main__":
    test_models()