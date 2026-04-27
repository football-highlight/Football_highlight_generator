"""
Audio feature extraction models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AudioCNN(nn.Module):
    """CNN for audio feature extraction"""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_channels: int = 1,
        dropout_rate: float = 0.5
    ):
        super(AudioCNN, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers for MFCC features (13 coefficients x time frames)
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Calculate output size
        # Input: (batch, 1, 13, 100)
        # After conv layers: (batch, 128, 1, 1)
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
        # Feature extraction layer
        self.feature_extractor = nn.Sequential(
            self.conv_layers,
            nn.Flatten()
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized AudioCNN with {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
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
            x: Input tensor of shape (batch, channels, features, time)
            
        Returns:
            output: Classification logits
            features: Extracted features before classification
        """
        # Extract features
        features = self.feature_extractor(x)
        
        # Classification
        output = self.fc_layers(features)
        
        return output, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.feature_extractor(x)
        return features


class AudioLSTM(nn.Module):
    """LSTM for audio sequence processing"""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 13,  # MFCC coefficients
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.5
    ):
        super(AudioLSTM, self).__init__()
        
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized AudioLSTM with {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize weights"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, features, time)
            
        Returns:
            output: Classification logits
            features: Extracted features before classification
        """
        # Reshape for LSTM: (batch, time, features)
        batch_size = x.size(0)
        
        # Squeeze channel dimension and transpose
        if x.dim() == 4:  # (batch, 1, features, time)
            x = x.squeeze(1)  # (batch, features, time)
        
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(x)  # lstm_out: (batch, time, hidden_size*2)
        
        # Apply attention
        attention_weights = self.attention(lstm_out)  # (batch, time, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden_size*2)
        
        # Classification
        output = self.classifier(context)
        
        return output, context


class AudioTransformer(nn.Module):
    """Transformer for audio feature processing"""
    
    def __init__(
        self,
        num_classes: int = 10,
        input_dim: int = 13,
        model_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout_rate: float = 0.1
    ):
        super(AudioTransformer, self).__init__()
        
        self.num_classes = num_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, model_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(model_dim, dropout_rate)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(model_dim // 2, num_classes)
        )
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        logger.info(f"Initialized AudioTransformer with {num_classes} classes")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, channels, features, time)
            
        Returns:
            output: Classification logits
            features: Extracted features before classification
        """
        batch_size = x.size(0)
        
        # Reshape: (batch, time, features)
        if x.dim() == 4:  # (batch, 1, features, time)
            x = x.squeeze(1)  # (batch, features, time)
        
        x = x.transpose(1, 2)  # (batch, time, features)
        
        # Input projection
        x = self.input_projection(x)  # (batch, time, model_dim)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(x)  # (batch, time, model_dim)
        
        # Global average pooling
        pooled = self.pooling(transformer_out.transpose(1, 2))  # (batch, model_dim, 1)
        features = pooled.squeeze(2)  # (batch, model_dim)
        
        # Classification
        output = self.classifier(features)
        
        return output, features


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def create_audio_model(
    model_type: str = 'cnn',
    num_classes: int = 10,
    **kwargs
) -> nn.Module:
    """
    Factory function to create audio models
    
    Args:
        model_type: Type of model to create
        num_classes: Number of output classes
        **kwargs: Additional model-specific parameters
        
    Returns:
        Initialized model
    """
    model_registry = {
        'cnn': AudioCNN,
        'lstm': AudioLSTM,
        'transformer': AudioTransformer,
    }
    
    if model_type not in model_registry:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available types: {list(model_registry.keys())}")
    
    model_class = model_registry[model_type]
    model = model_class(num_classes=num_classes, **kwargs)
    
    logger.info(f"Created {model_type} audio model with {num_classes} classes")
    
    return model


def test_audio_models():
    """Test audio model functions"""
    print("Testing audio models...")
    
    # Test AudioCNN
    print("\n1. Testing AudioCNN...")
    model = AudioCNN(num_classes=10)
    
    dummy_input = torch.randn(2, 1, 13, 100)
    output, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print("✅ AudioCNN test passed")
    
    # Test AudioLSTM
    print("\n2. Testing AudioLSTM...")
    model = AudioLSTM(num_classes=10)
    
    output, features = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Features shape: {features.shape}")
    print("✅ AudioLSTM test passed")
    
    # Test create_audio_model
    print("\n3. Testing create_audio_model...")
    model = create_audio_model(
        model_type='cnn',
        num_classes=10
    )
    
    print(f"✅ Created model: {model.__class__.__name__}")
    
    print("\n✅ All audio model tests passed!")


if __name__ == "__main__":
    test_audio_models()