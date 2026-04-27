"""
Model training pipeline with comprehensive metrics tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
import json
import os
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from config.config import config
from src.utils.logger import setup_logger, log_metrics
from src.evaluation.metrics_calculator import ModelEvaluator

logger = setup_logger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        if mode == 'min':
            self.best_score = float('inf')
            self.compare = lambda x, y: x < y - min_delta
        else:  # mode == 'max'
            self.best_score = float('-inf')
            self.compare = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop early
        
        Args:
            score: Current validation score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def reset(self):
        """Reset early stopping counter"""
        self.counter = 0
        self.early_stop = False


class ModelTrainer:
    """Complete training pipeline with metrics tracking"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        experiment_name: str = None,
        use_wandb: bool = False,
        device: str = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Device configuration
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Experiment tracking
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir = os.path.join('experiments', self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Subdirectories
        self.checkpoints_dir = os.path.join(self.experiment_dir, 'checkpoints')
        self.plots_dir = os.path.join(self.experiment_dir, 'plots')
        self.logs_dir = os.path.join(self.experiment_dir, 'logs')
        
        for dir_path in [self.checkpoints_dir, self.plots_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            self._init_wandb()
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = None
        self.best_epoch = 0
        
        # Metrics tracking
        self.train_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        self.val_history = {
            'loss': [], 'accuracy': [], 'precision': [], 'recall': [], 'f1': []
        }
        
        # Initialize components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.early_stopping = EarlyStopping(
            patience=config.MODEL.PATIENCE,
            min_delta=config.MODEL.MIN_DELTA,
            mode='min'
        )
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Initialized ModelTrainer")
        logger.info(f"Device: {self.device}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
        logger.info(f"Number of parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config"""
        optimizer_type = config.MODEL.OPTIMIZER.lower()
        lr = config.MODEL.CNN_LEARNING_RATE
        weight_decay = config.MODEL.WEIGHT_DECAY
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = config.MODEL.SCHEDULER.lower()
        
        if scheduler_type == 'reduce_on_plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=config.MODEL.NUM_EPOCHS,
                eta_min=1e-6,
                verbose=True
            )
        else:
            return None
    
    def _init_wandb(self):
        """Initialize Weights & Biases"""
        try:
            wandb.init(
                project="football-highlights",
                name=self.experiment_name,
                config={
                    'video_config': config.VIDEO.__dict__,
                    'model_config': config.MODEL.__dict__,
                    'event_config': config.EVENT.__dict__,
                },
                dir=self.experiment_dir
            )
            logger.info("Weights & Biases initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.use_wandb = False
    
    def _extract_batch_data(self, batch):
        """Extract video, audio, and labels from batch with format handling"""
        if isinstance(batch, dict):
            # Dictionary format (expected)
            video = batch['video'].to(self.device)
            audio = batch['audio'].to(self.device)
            # Note: In your original code it's 'label' not 'labels'
            labels_key = 'label' if 'label' in batch else 'labels'
            labels = batch[labels_key].to(self.device)
        elif isinstance(batch, (list, tuple)):
            # List/tuple format (fallback for dummy data)
            # Assuming format: [video, audio, labels] or (video, audio, labels)
            if len(batch) >= 3:
                video = batch[0].to(self.device)
                audio = batch[1].to(self.device)
                labels = batch[2].to(self.device)
            else:
                raise ValueError(f"Batch has insufficient elements. Expected 3, got {len(batch)}")
        else:
            raise ValueError(f"Unknown batch type: {type(batch)}. Expected dict or list/tuple.")
        
        return video, audio, labels
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1} [Train]",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Extract batch data with format handling
            video, audio, labels = self._extract_batch_data(batch)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, 'forward_multimodal'):
                outputs = self.model.forward_multimodal(video, audio)
            else:
                outputs = self.model(video, audio)
                
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if config.MODEL.GRADIENT_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    config.MODEL.GRADIENT_CLIP
                )
                
            self.optimizer.step()
            
            # Calculate metrics
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            batch_acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': f'{batch_acc:.2%}'
            })
            
            # Log batch metrics to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_loss': loss.item(),
                    'train/batch_accuracy': batch_acc,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Calculate epoch metrics
        epoch_loss = total_loss / len(self.train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1
        }
        
        # Update history
        for key in metrics:
            self.train_history[key].append(metrics[key])
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f'train/{k}': v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=self.current_epoch)
        
        return metrics
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader,
                desc=f"Epoch {self.current_epoch + 1} [Validation]",
                leave=False
            )
            
            for batch in progress_bar:
                # Extract batch data with format handling
                video, audio, labels = self._extract_batch_data(batch)
                
                # Forward pass
                if hasattr(self.model, 'forward_multimodal'):
                    outputs = self.model.forward_multimodal(video, audio)
                else:
                    outputs = self.model(video, audio)
                    
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': loss.item()
                })
        
        # Calculate metrics
        epoch_loss = total_loss / len(self.val_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # Calculate AUC-ROC if possible
        try:
            auc_roc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
        except:
            auc_roc = 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': epoch_precision,
            'recall': epoch_recall,
            'f1': epoch_f1,
            'auc_roc': auc_roc
        }
        
        # Update history
        for key in metrics:
            if key in self.val_history:
                self.val_history[key].append(metrics[key])
        
        # Log to wandb
        if self.use_wandb:
            wandb_metrics = {f'val/{k}': v for k, v in metrics.items()}
            wandb.log(wandb_metrics, step=self.current_epoch)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'config': config.__dict__
        }
        
        # Save latest checkpoint
        latest_path = os.path.join(self.checkpoints_dir, 'latest.pth')
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = os.path.join(self.checkpoints_dir, f'epoch_{self.current_epoch:03d}.pth')
        torch.save(checkpoint, epoch_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, 'best.pth')
            torch.save(checkpoint, best_path)
            
            # Save best metrics
            metrics_path = os.path.join(self.experiment_dir, 'best_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.best_metrics, f, indent=2)
    
    def train(self, num_epochs: int = None) -> Dict[str, Any]:
        """
        Main training loop
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training history and best model path
        """
        num_epochs = num_epochs or config.MODEL.NUM_EPOCHS
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {self.current_epoch}/{num_epochs}")
            logger.info(f"{'='*60}")
            
            # Training phase
            train_metrics = self.train_epoch()
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.2%}, "
                       f"F1: {train_metrics['f1']:.4f}")
            
            # Validation phase
            val_metrics = self.validate()
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.2%}, "
                       f"F1: {val_metrics['f1']:.4f}")
            
            # Update learning rate
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # Save checkpoint
            is_best = False
            if self.best_metrics is None or val_metrics['loss'] < self.best_metrics['loss']:
                self.best_metrics = val_metrics.copy()
                self.best_epoch = self.current_epoch
                is_best = True
                logger.info(f"🎯 New best model! Loss: {val_metrics['loss']:.4f}")
            
            self.save_checkpoint(is_best=is_best)
            
            # Check early stopping
            if self.early_stopping(val_metrics['loss']):
                logger.info(f"⏹️ Early stopping triggered at epoch {self.current_epoch}")
                break
        
        # Finalize training
        logger.info(f"\n{'='*60}")
        logger.info("Training completed!")
        logger.info(f"{'='*60}")
        logger.info(f"Best model at epoch {self.best_epoch}:")
        for metric, value in self.best_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Save final training history
        self._save_training_history()
        
        # Test the best model
        logger.info("\nEvaluating on test set...")
        test_results = self.evaluate_on_test()
        
        # Log final results
        log_metrics(test_results, "Test Set Results", logger)
        
        # Save experiment summary
        self._save_experiment_summary(test_results)
        
        # Close wandb if used
        if self.use_wandb:
            wandb.finish()
        
        return {
            'train_history': self.train_history,
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'best_epoch': self.best_epoch,
            'test_results': test_results,
            'best_model_path': os.path.join(self.checkpoints_dir, 'best.pth'),
            'experiment_dir': self.experiment_dir
        }
    
    def evaluate_on_test(self) -> Dict[str, float]:
        """
        Evaluate model on test set
        
        Returns:
            Dictionary of test metrics
        """
        # Load best model
        best_path = os.path.join(self.checkpoints_dir, 'best.pth')
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        test_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.test_loader,
                desc="Evaluating on Test Set",
                leave=False
            )
            
            for batch in progress_bar:
                # Extract batch data with format handling
                video, audio, labels = self._extract_batch_data(batch)
                
                # Forward pass
                if hasattr(self.model, 'forward_multimodal'):
                    outputs = self.model.forward_multimodal(video, audio)
                else:
                    outputs = self.model(video, audio)
                    
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = self.evaluator.calculate_all_metrics(
            all_labels, all_preds, all_probs
        )
        metrics['loss'] = test_loss / len(self.test_loader)
        
        # Generate detailed classification report
        class_names = getattr(config.EVENT, 'CLASS_NAMES', None)
        if class_names is None and hasattr(self.test_loader.dataset, 'classes'):
            class_names = self.test_loader.dataset.classes
        
        # Classification report
        logger.info("\n" + "="*60)
        logger.info("Classification Report:")
        logger.info("="*60)
        
        report = classification_report(
            all_labels, 
            all_preds, 
            target_names=class_names,
            digits=4
        )
        logger.info(f"\n{report}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        # Save detailed results
        self._save_test_results(
            all_labels, all_preds, all_probs, cm, report, metrics
        )
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({f'test/{k}': v for k, v in metrics.items()})
            
            # Log confusion matrix
            if class_names:
                wandb.log({
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_labels,
                        preds=all_preds,
                        class_names=class_names
                    )
                })
        
        return metrics
    
    def _save_training_history(self):
        """Save training history to file"""
        history = {
            'train': self.train_history,
            'val': self.val_history,
            'best_epoch': self.best_epoch,
            'best_metrics': self.best_metrics,
            'config': config.__dict__
        }
        
        history_path = os.path.join(self.experiment_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        logger.info(f"Training history saved to {history_path}")
    
    def _save_experiment_summary(self, test_results: Dict[str, float]):
        """Save experiment summary"""
        summary = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'model_name': self.model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in self.model.parameters()),
            'training_epochs': self.current_epoch,
            'best_epoch': self.best_epoch,
            'best_validation_metrics': self.best_metrics,
            'test_metrics': test_results,
            'config': {
                'video': config.VIDEO.__dict__,
                'model': config.MODEL.__dict__,
                'event': config.EVENT.__dict__,
            }
        }
        
        summary_path = os.path.join(self.experiment_dir, 'experiment_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Experiment summary saved to {summary_path}")
    
    def _save_test_results(
        self,
        labels: List[int],
        preds: List[int],
        probs: List[np.ndarray],
        cm: np.ndarray,
        report: str,
        metrics: Dict[str, float]
    ):
        """Save detailed test results"""
        results = {
            'metrics': metrics,
            'predictions': preds,
            'labels': labels,
            'probabilities': [prob.tolist() for prob in probs],
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = os.path.join(self.experiment_dir, 'test_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save confusion matrix as CSV
        cm_path = os.path.join(self.plots_dir, 'confusion_matrix.csv')
        np.savetxt(cm_path, cm, delimiter=',', fmt='%d')
        
        logger.info(f"Detailed test results saved to {results_path}")
    
    def plot_training_history(self, save: bool = True):
        """
        Plot training history
        
        Args:
            save: Whether to save the plots to file
        """
        try:
            import matplotlib.pyplot as plt
            
            metrics_to_plot = ['loss', 'accuracy', 'f1']
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                
                if metric in self.train_history and len(self.train_history[metric]) > 0:
                    epochs = range(1, len(self.train_history[metric]) + 1)
                    ax.plot(epochs, self.train_history[metric], 'b-', label='Train')
                    
                if metric in self.val_history and len(self.val_history[metric]) > 0:
                    epochs = range(1, len(self.val_history[metric]) + 1)
                    ax.plot(epochs, self.val_history[metric], 'r-', label='Validation')
                
                ax.set_title(f'{metric.capitalize()} over Epochs')
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Mark best epoch
                if self.best_epoch > 0:
                    if metric in self.val_history and len(self.val_history[metric]) >= self.best_epoch:
                        best_value = self.val_history[metric][self.best_epoch - 1]
                        ax.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.5)
                        ax.plot(self.best_epoch, best_value, 'go', label=f'Best ({self.best_epoch})')
                        ax.legend()
            
            plt.tight_layout()
            
            if save:
                plot_path = os.path.join(self.plots_dir, 'training_history.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                logger.info(f"Training history plot saved to {plot_path}")
                
                # Save to wandb
                if self.use_wandb:
                    wandb.log({"training_plots/history": wandb.Image(plot_path)})
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Skipping plots.")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        if checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training history
        self.train_history = checkpoint.get('train_history', self.train_history)
        self.val_history = checkpoint.get('val_history', self.val_history)
        self.best_metrics = checkpoint.get('best_metrics', self.best_metrics)
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_epoch = self.current_epoch
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")
        
        if self.best_metrics:
            logger.info("Best metrics from checkpoint:")
            for metric, value in self.best_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")


def create_trainer(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    experiment_name: str = None,
    use_wandb: bool = False,
    num_epochs: int = None
) -> Dict[str, Any]:
    """
    Convenience function to train a model
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        experiment_name: Name for the experiment
        use_wandb: Whether to use Weights & Biases
        num_epochs: Number of epochs to train
        
    Returns:
        Training results dictionary
    """
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        experiment_name=experiment_name,
        use_wandb=use_wandb
    )
    
    results = trainer.train(num_epochs=num_epochs)
    
    # Plot training history
    trainer.plot_training_history(save=True)
    
    return results