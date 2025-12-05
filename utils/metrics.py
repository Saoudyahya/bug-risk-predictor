"""
Evaluation metrics for bug prediction models
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, matthews_corrcoef,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Calculate and visualize evaluation metrics for bug prediction
    """
    
    def __init__(self):
        """Initialize the metrics calculator"""
        self.metrics = {}
    
    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate all relevant metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Add AUC if probabilities are provided
        if y_proba is not None:
            try:
                if len(y_proba.shape) == 2:
                    # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except Exception as e:
                print(f"Could not calculate ROC AUC: {e}")
        
        self.metrics = metrics
        return metrics
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """
        Calculate confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        target_names: Optional[list] = None
    ) -> str:
        """
        Get classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Classification report as string
        """
        if target_names is None:
            target_names = ['Clean', 'Buggy']
        
        return classification_report(
            y_true, y_pred,
            target_names=target_names,
            zero_division=0
        )
    
    def print_metrics(self, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Print metrics in a formatted way
        
        Args:
            metrics: Dictionary of metrics (uses stored metrics if None)
        """
        if metrics is None:
            metrics = self.metrics
        
        print("\n" + "="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        
        for metric_name, value in metrics.items():
            print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
        
        print("="*50 + "\n")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[list] = None,
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
        """
        if labels is None:
            labels = ['Clean', 'Buggy']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar=True
        )
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "ROC Curve"
    ) -> None:
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
        """
        # Handle 2D probability array
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        title: str = "Precision-Recall Curve"
    ) -> None:
        """
        Plot precision-recall curve
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            title: Plot title
        """
        # Handle 2D probability array
        if len(y_proba.shape) == 2:
            y_proba = y_proba[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(
        self,
        feature_names: list,
        importance_scores: np.ndarray,
        top_n: int = 15,
        title: str = "Feature Importance"
    ) -> None:
        """
        Plot feature importance
        
        Args:
            feature_names: List of feature names
            importance_scores: Importance scores
            top_n: Number of top features to display
            title: Plot title
        """
        # Sort by importance
        indices = np.argsort(importance_scores)[::-1][:top_n]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(top_n),
            importance_scores[indices],
            align='center'
        )
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def compare_models(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = 'f1_score'
    ) -> None:
        """
        Compare multiple models
        
        Args:
            results: Dictionary mapping model names to their metrics
            metric: Metric to compare
        """
        model_names = list(results.keys())
        metric_values = [results[model][metric] for model in model_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(model_names, metric_values)
        
        # Color bars
        colors = plt.cm.viridis(np.linspace(0, 1, len(model_names)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}',
                  fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def create_metrics_comparison_table(
        self,
        results: Dict[str, Dict[str, float]]
    ) -> pd.DataFrame:
        """
        Create a comparison table of model metrics
        
        Args:
            results: Dictionary mapping model names to their metrics
            
        Returns:
            DataFrame with comparison
        """
        df = pd.DataFrame(results).T
        
        # Round values
        df = df.round(4)
        
        # Highlight best values
        styled_df = df.style.highlight_max(axis=0, color='lightgreen')
        
        return df
    
    def calculate_cost_sensitive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        fp_cost: float = 1.0,
        fn_cost: float = 10.0
    ) -> Dict[str, float]:
        """
        Calculate cost-sensitive metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            fp_cost: Cost of false positive
            fn_cost: Cost of false negative
            
        Returns:
            Dictionary with cost metrics
        """
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        total_cost = fp * fp_cost + fn * fn_cost
        avg_cost = total_cost / len(y_true)
        
        return {
            'total_cost': total_cost,
            'average_cost': avg_cost,
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'fp_cost_total': fp * fp_cost,
            'fn_cost_total': fn * fn_cost
        }


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    feature_names: Optional[list] = None,
    feature_importance: Optional[np.ndarray] = None,
    model_name: str = "Model",
    plot: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        feature_names: List of feature names
        feature_importance: Feature importance scores
        model_name: Name of the model
        plot: Whether to create plots
        
    Returns:
        Dictionary with all evaluation results
    """
    calc = MetricsCalculator()
    
    # Calculate metrics
    metrics = calc.calculate_all_metrics(y_true, y_pred, y_proba)
    
    # Print metrics
    print(f"\n{'='*50}")
    print(f"EVALUATION: {model_name}")
    print(f"{'='*50}")
    calc.print_metrics(metrics)
    
    # Print classification report
    print("\nClassification Report:")
    print(calc.get_classification_report(y_true, y_pred))
    
    # Create plots if requested
    if plot:
        calc.plot_confusion_matrix(y_true, y_pred, title=f"{model_name} - Confusion Matrix")
        
        if y_proba is not None:
            calc.plot_roc_curve(y_true, y_proba, title=f"{model_name} - ROC Curve")
            calc.plot_precision_recall_curve(y_true, y_proba, title=f"{model_name} - PR Curve")
        
        if feature_importance is not None and feature_names is not None:
            calc.plot_feature_importance(
                feature_names,
                feature_importance,
                title=f"{model_name} - Feature Importance"
            )
    
    return {
        'metrics': metrics,
        'confusion_matrix': calc.get_confusion_matrix(y_true, y_pred),
        'classification_report': calc.get_classification_report(y_true, y_pred)
    }


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Generate sample predictions
    y_true = np.random.randint(0, 2, 200)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(len(y_true), 30, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_proba = np.random.rand(200, 2)
    y_proba /= y_proba.sum(axis=1, keepdims=True)
    
    # Evaluate
    results = evaluate_model(
        y_true, y_pred, y_proba,
        model_name="Example Model",
        plot=False
    )
    
    print("\nEvaluation complete!")
