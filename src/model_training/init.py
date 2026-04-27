"""
Model training module for football highlights generation
"""

from .trainer import ModelTrainer, EarlyStopping, create_trainer

__all__ = [
    "ModelTrainer",
    "EarlyStopping",
    "create_trainer"
]