"""
Comprehensive metrics calculation for model evaluation
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, cohen_kappa_score, hamming_loss,
    jaccard_score, log_loss
)
from typing import Dict, List, Tuple, Optional, Any
import json
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation with multiple metrics"""
    
    def __init__(self, num_classes: int = None):
        self.num_classes = num_classes
    
    def calculate_all_metrics(
        self, 
        y_true: List, 
        y_pred: List, 
        y_prob: Optional[np.ndarray] = None,
        average: str = 'weighted'
    ) -> Dict[str, Any]:
        """
        Calculate all evaluation metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            average: Averaging method for multi-class metrics
            
        Returns:
            Dictionary of all metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # Additional classification metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        metrics['cohens_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['hamming_loss'] = hamming_loss(y_true, y_pred)
        
        # Jaccard score (IoU)
        metrics['jaccard_score'] = jaccard_score(y_true, y_pred, average=average)
        
        # Per-class metrics
        unique_classes = np.unique(y_true)
        per_class_metrics = {}
        
        for cls in unique_classes:
            # Binary classification for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            if len(np.unique(y_true_binary)) > 1:  # Check if class exists in test set
                try:
                    per_class_metrics[int(cls)] = {
                        'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                        'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                        'f1_score': f1_score(y_true_binary, y_pred_binary, zero_division=0),
                        'support': np.sum(y_true_binary)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate metrics for class {cls}: {e}")
        
        metrics['per_class'] = per_class_metrics
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
        
        # Advanced metrics if probabilities are available
        if y_prob is not None:
            y_prob = np.array(y_prob)
            
            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except:
                metrics['log_loss'] = float('inf')
            
            try:
                metrics['auc_roc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=average
                )
            except:
                metrics['auc_roc'] = 0.0
            
            try:
                metrics['average_precision'] = average_precision_score(
                    y_true, y_prob, average=average
                )
            except:
                metrics['average_precision'] = 0.0
            
            # Precision-Recall curve data
            if len(unique_classes) == 2:  # Binary classification
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob[:, 1])
                metrics['precision_recall_curve'] = {
                    'precision': precision_curve.tolist(),
                    'recall': recall_curve.tolist()
                }
        
        # Statistical metrics
        metrics = self._calculate_statistical_metrics(metrics, y_true, y_pred)
        
        # Summary metrics
        metrics['overall'] = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'matthews_corrcoef': metrics['matthews_corrcoef'],
            'cohens_kappa': metrics['cohens_kappa']
        }
        
        if 'auc_roc' in metrics:
            metrics['overall']['auc_roc'] = metrics['auc_roc']
        
        logger.info(f"Calculated {len(metrics)} metrics")
        
        return metrics
    
    def _calculate_statistical_metrics(
        self, 
        metrics: Dict, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict:
        """Calculate additional statistical metrics"""
        
        # Error analysis
        errors = y_true != y_pred
        error_rate = np.mean(errors) * 100
        metrics['error_rate'] = error_rate
        metrics['error_count'] = int(np.sum(errors))
        metrics['correct_count'] = int(np.sum(~errors))
        
        # Most confused classes
        cm = np.array(metrics['confusion_matrix'])
        n_classes = cm.shape[0]
        
        confusion_pairs = []
        for i in range(n_classes):
            for j in range(n_classes):
                if i != j and cm[i, j] > 0:
                    confusion_pairs.append({
                        'true_class': int(i),
                        'predicted_class': int(j),
                        'count': int(cm[i, j]),
                        'percentage': cm[i, j] / np.sum(cm[i, :]) * 100 if np.sum(cm[i, :]) > 0 else 0
                    })
        
        # Sort by count (descending)
        confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
        metrics['top_confusions'] = confusion_pairs[:10]  # Top 10 confusions
        
        # Class-wise metrics summary
        class_stats = []
        for cls, stats in metrics['per_class'].items():
            class_stats.append({
                'class': int(cls),
                'precision': stats['precision'],
                'recall': stats['recall'],
                'f1_score': stats['f1_score'],
                'support': stats['support']
            })
        
        metrics['class_statistics'] = class_stats
        
        # Calculate macro and micro averages
        if class_stats:
            precisions = [s['precision'] for s in class_stats]
            recalls = [s['recall'] for s in class_stats]
            f1_scores = [s['f1_score'] for s in class_stats]
            
            metrics['macro_precision'] = np.mean(precisions)
            metrics['macro_recall'] = np.mean(recalls)
            metrics['macro_f1'] = np.mean(f1_scores)
            
            # Calculate standard deviations
            metrics['std_precision'] = np.std(precisions)
            metrics['std_recall'] = np.std(recalls)
            metrics['std_f1'] = np.std(f1_scores)
        
        # Calculate per-class accuracy
        per_class_accuracy = []
        for i in range(n_classes):
            class_mask = y_true == i
            if np.any(class_mask):
                class_accuracy = np.mean(y_pred[class_mask] == i)
                per_class_accuracy.append(class_accuracy)
        
        if per_class_accuracy:
            metrics['per_class_accuracy'] = per_class_accuracy
            metrics['mean_class_accuracy'] = np.mean(per_class_accuracy)
            metrics['std_class_accuracy'] = np.std(per_class_accuracy)
        
        return metrics
    
    def calculate_confidence_metrics(
        self, 
        y_true: List, 
        y_prob: np.ndarray,
        confidence_thresholds: List[float] = None
    ) -> Dict:
        """
        Calculate metrics at different confidence thresholds
        
        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            confidence_thresholds: List of confidence thresholds
            
        Returns:
            Dictionary of confidence-based metrics
        """
        if confidence_thresholds is None:
            confidence_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        metrics_by_threshold = {}
        
        for threshold in confidence_thresholds:
            # Get predictions above threshold
            max_probs = np.max(y_prob, axis=1)
            confident_mask = max_probs >= threshold
            
            if np.any(confident_mask):
                y_true_confident = np.array(y_true)[confident_mask]
                y_pred_confident = np.argmax(y_prob[confident_mask], axis=1)
                
                threshold_metrics = {
                    'samples_above_threshold': int(np.sum(confident_mask)),
                    'percentage_above_threshold': np.mean(confident_mask) * 100,
                    'accuracy': accuracy_score(y_true_confident, y_pred_confident),
                    'precision': precision_score(y_true_confident, y_pred_confident, average='weighted', zero_division=0),
                    'recall': recall_score(y_true_confident, y_pred_confident, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true_confident, y_pred_confident, average='weighted', zero_division=0)
                }
            else:
                threshold_metrics = {
                    'samples_above_threshold': 0,
                    'percentage_above_threshold': 0,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0
                }
            
            metrics_by_threshold[threshold] = threshold_metrics
        
        return metrics_by_threshold
    
    def compare_models(
        self, 
        model_results: Dict[str, Dict],
        primary_metric: str = 'accuracy'
    ) -> Dict:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary of model names to their results
            primary_metric: Primary metric for comparison
            
        Returns:
            Comparison results
        """
        comparison = {
            'models': {},
            'best_model': None,
            'best_score': float('-inf'),
            'summary': {},
            'rankings': {}
        }
        
        # Collect metrics for each model
        for model_name, results in model_results.items():
            overall = results.get('overall', {})
            
            comparison['models'][model_name] = {
                'accuracy': overall.get('accuracy', 0),
                'precision': overall.get('precision', 0),
                'recall': overall.get('recall', 0),
                'f1_score': overall.get('f1_score', 0),
                'matthews_corrcoef': overall.get('matthews_corrcoef', 0),
                'error_rate': results.get('error_rate', 0)
            }
            
            # Update best model
            primary_score = overall.get(primary_metric, 0)
            if primary_score > comparison['best_score']:
                comparison['best_score'] = primary_score
                comparison['best_model'] = model_name
        
        # Calculate rankings for each metric
        metrics_to_rank = ['accuracy', 'precision', 'recall', 'f1_score', 'matthews_corrcoef']
        
        for metric in metrics_to_rank:
            model_scores = []
            for model_name, model_metrics in comparison['models'].items():
                model_scores.append((model_name, model_metrics.get(metric, 0)))
            
            # Sort by score (descending)
            model_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Create ranking
            ranking = {}
            for rank, (model_name, score) in enumerate(model_scores, 1):
                ranking[model_name] = {
                    'rank': rank,
                    'score': score
                }
            
            comparison['rankings'][metric] = ranking
        
        # Calculate relative improvements
        baseline_model = None
        for model_name in comparison['models']:
            if baseline_model is None:
                baseline_model = model_name
                comparison['summary'][model_name] = 'Baseline'
            else:
                baseline_score = comparison['models'][baseline_model]['accuracy']
                model_score = comparison['models'][model_name]['accuracy']
                
                if baseline_score > 0:
                    improvement = ((model_score - baseline_score) / baseline_score) * 100
                    comparison['summary'][model_name] = f"{improvement:+.2f}% improvement"
                else:
                    comparison['summary'][model_name] = "N/A (baseline score is 0)"
        
        return comparison
    
    def save_evaluation_report(
        self, 
        metrics: Dict, 
        output_path: str,
        model_info: Optional[Dict] = None
    ) -> str:
        """
        Save comprehensive evaluation report
        
        Args:
            metrics: Evaluation metrics
            output_path: Path to save report
            model_info: Additional model information
            
        Returns:
            Path to saved report
        """
        report = {
            'evaluation_metadata': {
                'timestamp': self._get_timestamp(),
                'evaluator': 'ModelEvaluator',
                'version': '1.0'
            },
            'model_info': model_info or {},
            'metrics_summary': {
                'overall_accuracy': metrics.get('accuracy', 0),
                'overall_f1_score': metrics.get('f1_score', 0),
                'error_rate': metrics.get('error_rate', 0),
                'matthews_corrcoef': metrics.get('matthews_corrcoef', 0),
                'cohens_kappa': metrics.get('cohens_kappa', 0)
            },
            'detailed_metrics': metrics,
            'recommendations': self._generate_recommendations(metrics)
        }
        
        # Add advanced metrics if available
        if 'auc_roc' in metrics:
            report['metrics_summary']['auc_roc'] = metrics['auc_roc']
        
        if 'average_precision' in metrics:
            report['metrics_summary']['average_precision'] = metrics['average_precision']
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved evaluation report to {output_path}")
        
        # Also generate markdown report
        md_report = self._generate_markdown_report(report)
        md_path = output_path.replace('.json', '.md')
        with open(md_path, 'w') as f:
            f.write(md_report)
        
        return output_path
    
    def _generate_recommendations(self, metrics: Dict) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Check overall accuracy
        accuracy = metrics.get('accuracy', 0)
        if accuracy < 0.7:
            recommendations.append(
                f"Model accuracy ({accuracy:.2%}) is below 70%. Consider collecting more "
                "training data or using data augmentation techniques."
            )
        
        # Check class imbalance
        class_stats = metrics.get('class_statistics', [])
        if class_stats:
            f1_scores = [s['f1_score'] for s in class_stats]
            supports = [s['support'] for s in class_stats]
            
            # Check for class imbalance in support
            if max(supports) / min(supports) > 10:
                recommendations.append(
                    "Severe class imbalance detected. Consider using class weights "
                    "or oversampling techniques for underrepresented classes."
                )
            
            # Check performance variance
            if max(f1_scores) - min(f1_scores) > 0.3:
                recommendations.append(
                    "Significant performance variance between classes. Some classes "
                    "have much lower performance than others."
                )
        
        # Check confusion patterns
        top_confusions = metrics.get('top_confusions', [])
        if top_confusions:
            for conf in top_confusions[:3]:  # Top 3 confusions
                if conf['percentage'] > 20:
                    recommendations.append(
                        f"High confusion ({conf['percentage']:.1f}%) between class "
                        f"{conf['true_class']} and {conf['predicted_class']}. "
                        "These classes might need more distinctive features."
                    )
        
        # Check per-class performance
        for cls_stats in class_stats:
            if cls_stats['f1_score'] < 0.5:
                recommendations.append(
                    f"Class {cls_stats['class']} has low F1-score ({cls_stats['f1_score']:.3f}). "
                    "Consider collecting more samples for this class."
                )
        
        # Check confidence calibration
        if 'log_loss' in metrics and metrics['log_loss'] > 1.0:
            recommendations.append(
                f"High log loss ({metrics['log_loss']:.3f}) suggests poor confidence calibration. "
                "Consider using calibration techniques like Platt scaling or isotonic regression."
            )
        
        return recommendations
    
    def _generate_markdown_report(self, report: Dict) -> str:
        """Generate markdown version of evaluation report"""
        
        md = f"""# Model Evaluation Report

## Summary
- **Overall Accuracy**: {report['metrics_summary']['overall_accuracy']:.4f}
- **F1 Score**: {report['metrics_summary']['overall_f1_score']:.4f}
- **Error Rate**: {report['metrics_summary']['error_rate']:.2f}%
- **Matthews Correlation Coefficient**: {report['metrics_summary']['matthews_corrcoef']:.4f}
- **Cohen's Kappa**: {report['metrics_summary']['cohens_kappa']:.4f}
"""
        
        # Add advanced metrics if available
        if 'auc_roc' in report['metrics_summary']:
            md += f"- **AUC-ROC**: {report['metrics_summary']['auc_roc']:.4f}\n"
        
        if 'average_precision' in report['metrics_summary']:
            md += f"- **Average Precision**: {report['metrics_summary']['average_precision']:.4f}\n"
        
        # Add detailed metrics section
        md += "\n## Detailed Metrics\n\n"
        
        # Add per-class metrics table
        class_stats = report['detailed_metrics'].get('class_statistics', [])
        if class_stats:
            md += "### Per-Class Performance\n"
            md += "| Class | Precision | Recall | F1-Score | Support |\n"
            md += "|-------|-----------|--------|----------|---------|\n"
            
            for stats in class_stats:
                md += f"| {stats['class']} | {stats['precision']:.4f} | "
                md += f"{stats['recall']:.4f} | {stats['f1_score']:.4f} | "
                md += f"{stats['support']} |\n"
        
        # Add confusion analysis
        top_confusions = report['detailed_metrics'].get('top_confusions', [])
        if top_confusions:
            md += "\n### Top Confusions\n"
            for conf in top_confusions[:5]:  # Top 5 confusions
                md += f"- Class {conf['true_class']} → Class {conf['predicted_class']}: "
                md += f"{conf['count']} samples ({conf['percentage']:.1f}%)\n"
        
        # Add recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            md += "\n## Recommendations\n"
            for i, rec in enumerate(recommendations, 1):
                md += f"{i}. {rec}\n"
        
        return md
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def test_evaluator():
    """Test evaluator functions"""
    print("Testing ModelEvaluator...")
    
    # Create dummy data
    y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 1, 2, 0, 1, 2, 0]  # One misclassification
    y_prob = np.array([
        [0.8, 0.1, 0.1],
        [0.1, 0.7, 0.2],
        [0.2, 0.6, 0.2],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8],
        [0.9, 0.05, 0.05],
        [0.1, 0.8, 0.1],
        [0.1, 0.2, 0.7],
        [0.8, 0.1, 0.1]
    ])
    
    # Create evaluator
    evaluator = ModelEvaluator(num_classes=3)
    
    # Test basic metrics
    print("1. Testing basic metrics calculation...")
    metrics = evaluator.calculate_all_metrics(y_true, y_pred, y_prob)
    
    print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
    print(f"✅ F1 Score: {metrics['f1_score']:.4f}")
    print(f"✅ Error Rate: {metrics['error_rate']:.2f}%")
    
    # Test confidence metrics
    print("\n2. Testing confidence metrics...")
    confidence_metrics = evaluator.calculate_confidence_metrics(y_true, y_prob)
    
    for threshold, thresh_metrics in confidence_metrics.items():
        print(f"  Threshold {threshold}: {thresh_metrics['samples_above_threshold']} samples, "
              f"Accuracy: {thresh_metrics['accuracy']:.4f}")
    
    # Test model comparison
    print("\n3. Testing model comparison...")
    model_results = {
        'model_a': metrics,
        'model_b': {**metrics, 'overall': {**metrics['overall'], 'accuracy': 0.95}}
    }
    
    comparison = evaluator.compare_models(model_results)
    print(f"✅ Best model: {comparison['best_model']}")
    
    # Test report generation
    print("\n4. Testing report generation...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
        
        model_info = {
            'name': 'test_model',
            'architecture': '3DCNN',
            'parameters': 1000000
        }
        
        report_path = evaluator.save_evaluation_report(metrics, temp_path, model_info)
        print(f"✅ Generated report at: {report_path}")
        
        # Cleanup
        os.unlink(temp_path)
        if os.path.exists(temp_path.replace('.json', '.md')):
            os.unlink(temp_path.replace('.json', '.md'))
    
    print("\n✅ All evaluator tests passed!")


if __name__ == "__main__":
    test_evaluator()