"""
Feature extraction module - MINIMAL VERSION
"""

# Import only what actually exists
try:
    from .cnn3d_model import Sports3DCNN
except ImportError:
    Sports3DCNN = None

try:
    from .audio_model import AudioCNN
except ImportError:
    AudioCNN = None

# Multimodal models - DON'T try to import MultiModal3DCNN
try:
    from .multimodal_fusion import (
        LightMultiModal3DCNN,
        SimpleMultimodalModel,
        create_multimodal_model,
        load_model
        # REMOVED: MultiModal3DCNN from import list
    )
    # Create the alias HERE instead of trying to import it
    MultiModal3DCNN = LightMultiModal3DCNN
except ImportError as e:
    print(f"Error importing multimodal components: {e}")
    LightMultiModal3DCNN = None
    SimpleMultimodalModel = None
    create_multimodal_model = None
    load_model = None
    MultiModal3DCNN = None

__all__ = [
    "Sports3DCNN",
    "AudioCNN",
    "MultiModal3DCNN",
    "LightMultiModal3DCNN",
    "SimpleMultimodalModel",
    "create_multimodal_model",
    "load_model"
]