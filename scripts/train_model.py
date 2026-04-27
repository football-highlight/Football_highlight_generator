#!/usr/bin/env python
"""
Train football event detection model
"""

import argparse
import os
import sys
from pathlib import Path
import logging
from datetime import datetime  # FIXED: Import datetime class directly

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from src.data_preparation.dataset_creator import load_dataset, create_data_loaders
from src.feature_extraction.multimodal_fusion import create_multimodal_model
from src.model_training.trainer import ModelTrainer, create_trainer
from src.visualization.training_visualizer import TrainingVisualizer
from config.config import config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_model(args):
    """Main training function"""
    
    logger.info("=" * 60)
    logger.info("FOOTBALL EVENT DETECTION MODEL TRAINING")
    logger.info("=" * 60)
    
    # Check if data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        logger.info("Note: Run preprocessing first: python scripts/preprocess_data.py")
    
    # Set device
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Load datasets
    logger.info("\n📊 Loading datasets...")
    
    try:
        train_dataset = load_dataset(
            args.data_dir,
            split='train',
            transform=None,
            max_samples=args.max_samples
        )
        
        # FIXED: Handle None max_samples
        val_samples = args.max_samples // 2 if args.max_samples else None
        val_dataset = load_dataset(
            args.data_dir,
            split='val',
            transform=None,
            max_samples=val_samples
        )
        
        test_samples = args.max_samples // 2 if args.max_samples else None
        test_dataset = load_dataset(
            args.data_dir,
            split='test',
            transform=None,
            max_samples=test_samples
        )
        
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        logger.info("Creating dummy datasets for testing...")
        
        # Create dummy datasets for testing
        from torch.utils.data import TensorDataset
        num_samples = min(100, args.max_samples or 100)
        
        dummy_video = torch.randn(num_samples, 3, 16, 224, 224)
        dummy_audio = torch.randn(num_samples, 1, 13, 100)
        dummy_labels = torch.randint(0, 10, (num_samples,))
        
        train_dataset = TensorDataset(dummy_video, dummy_audio, dummy_labels)
        val_dataset = TensorDataset(dummy_video, dummy_audio, dummy_labels)
        test_dataset = TensorDataset(dummy_video, dummy_audio, dummy_labels)
    
    logger.info(f"✅ Train samples: {len(train_dataset)}")
    logger.info(f"✅ Validation samples: {len(val_dataset)}")
    logger.info(f"✅ Test samples: {len(test_dataset)}")
    
    # Create data loaders
    logger.info("\n🔄 Creating data loaders...")
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=0
    )
    
    # Create model
    logger.info("\n🤖 Creating model...")
    
    num_classes = len(train_dataset.label_map) if hasattr(train_dataset, 'label_map') else 10
    
    try:
        model = create_multimodal_model(
            model_type='simple',
            num_classes=num_classes,
            pretrained=False,
            fusion_method='concat'
        )
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        logger.info("Creating simple model for testing...")
        
        # Create a simple model for testing
        from src.feature_extraction.multimodal_fusion import SimpleMultimodalModel
        model = SimpleMultimodalModel(num_classes=num_classes)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"✅ Model created: {model.__class__.__name__}")
    logger.info(f"✅ Number of classes: {num_classes}")
    logger.info(f"✅ Total parameters: {total_params:,}")
    logger.info(f"✅ Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    logger.info("\n🎯 Initializing trainer...")
    
    # FIXED: datetime.now() works now because we imported datetime class directly
    experiment_name = args.experiment_name or f"{args.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        experiment_name=experiment_name,
        use_wandb=args.use_wandb
    )
    
    # Train model
    logger.info("\n🚀 Starting training...")
    
    try:
        training_result = trainer.train(num_epochs=args.epochs)
        
        logger.info(f"\n✅ Training completed successfully!")
        logger.info(f"📁 Experiment directory: {training_result['experiment_dir']}")
        logger.info(f"💾 Best model saved: {training_result['best_model_path']}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Evaluate on test set
    if args.evaluate:
        logger.info("\n📈 Evaluating on test set...")
        
        try:
            # FIXED: Changed from evaluate_test_set() to evaluate_on_test()
            test_results = trainer.evaluate_on_test()
            
            logger.info(f"Test Results:")
            # FIXED: Different metric names in the trainer
            logger.info(f"  Accuracy: {test_results.get('accuracy', 0):.4f}")
            logger.info(f"  F1 Score: {test_results.get('f1', 0):.4f}")
            logger.info(f"  Precision: {test_results.get('precision', 0):.4f}")
            logger.info(f"  Recall: {test_results.get('recall', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate visualizations
    if args.visualize:
        logger.info("\n🎨 Generating visualizations...")
        
        try:
            # FIXED: Trainer already has plot_training_history method
            trainer.plot_training_history(save=True)
            
            # FIXED: Removed create_training_report() call - it doesn't exist
            # trainer.create_training_report()
            
            logger.info(f"📊 Visualizations saved to: {training_result['experiment_dir']}/plots")
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    # Save model for deployment
    if args.save_deployment:
        logger.info("\n💾 Saving model for deployment...")
        
        try:
            deployment_dir = Path("models/deployment")
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = deployment_dir / f"{experiment_name}.pth"
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'label_map': train_dataset.label_map if hasattr(train_dataset, 'label_map') else {},
                'num_classes': num_classes,
                'model_type': args.model_type,
                'test_metrics': test_results if 'test_results' in locals() else {}
            }, model_path)
            
            logger.info(f"✅ Deployment model saved: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to save deployment model: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Train football event detection model")
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/datasets",
        help="Directory containing processed datasets"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use (for testing)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_type",
        type=str,
        default="multimodal",
        choices=["3dcnn", "audio", "multimodal", "two_stream"],
        help="Type of model to train"
    )
    
    parser.add_argument(
        "--pretrained",
        action="store_true",
        default=True,
        help="Use pretrained weights"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name for the experiment"
    )
    
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for tracking"
    )
    
    # Output arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        default=True,
        help="Evaluate on test set after training"
    )
    
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=True,
        help="Generate visualizations"
    )
    
    parser.add_argument(
        "--save_deployment",
        action="store_true",
        default=True,
        help="Save model for deployment"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run training
    train_model(args)


if __name__ == "__main__":
    main()