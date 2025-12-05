"""
Tests for metrics module
"""

import pytest
import numpy as np
import pandas as pd
from utils.metrics import (
    MetricsCalculator,
    evaluate_model
)


@pytest.mark.unit
class TestMetricsCalculator:
    """Tests for MetricsCalculator class"""
    
    def test_init(self):
        """Test metrics calculator initialization"""
        calc = MetricsCalculator()
        assert calc.metrics == {}
    
    def test_calculate_all_metrics(self):
        """Test calculating all metrics"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'matthews_corrcoef' in metrics
        assert 'cohen_kappa' in metrics
    
    def test_calculate_metrics_with_proba(self):
        """Test calculating metrics with probabilities"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        y_proba = np.array([
            [0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.3, 0.7],
            [0.8, 0.2], [0.7, 0.3], [0.1, 0.9], [0.9, 0.1]
        ])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
    
    def test_perfect_prediction(self):
        """Test metrics for perfect predictions"""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = y_true.copy()
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0


@pytest.mark.unit
class TestConfusionMatrix:
    """Tests for confusion matrix"""
    
    def test_get_confusion_matrix(self):
        """Test getting confusion matrix"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        calc = MetricsCalculator()
        cm = calc.get_confusion_matrix(y_true, y_pred)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == len(y_true)
    
    def test_confusion_matrix_values(self):
        """Test confusion matrix values"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        calc = MetricsCalculator()
        cm = calc.get_confusion_matrix(y_true, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        assert tn == 1  # True negatives
        assert fp == 1  # False positives
        assert fn == 1  # False negatives
        assert tp == 1  # True positives


@pytest.mark.unit
class TestClassificationReport:
    """Tests for classification report"""
    
    def test_get_classification_report(self):
        """Test getting classification report"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        calc = MetricsCalculator()
        report = calc.get_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert 'precision' in report
        assert 'recall' in report
        assert 'f1-score' in report
    
    def test_classification_report_custom_names(self):
        """Test classification report with custom names"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        
        calc = MetricsCalculator()
        report = calc.get_classification_report(
            y_true, y_pred,
            target_names=['No Bug', 'Bug']
        )
        
        assert 'No Bug' in report
        assert 'Bug' in report


@pytest.mark.unit
class TestPrintMetrics:
    """Tests for printing metrics"""
    
    def test_print_metrics(self, capsys):
        """Test printing metrics"""
        metrics = {
            'accuracy': 0.85,
            'precision': 0.80,
            'recall': 0.75,
            'f1_score': 0.77
        }
        
        calc = MetricsCalculator()
        calc.print_metrics(metrics)
        
        captured = capsys.readouterr()
        assert "MODEL EVALUATION METRICS" in captured.out
        assert "Accuracy" in captured.out
        assert "0.8500" in captured.out


@pytest.mark.unit
class TestCostSensitiveMetrics:
    """Tests for cost-sensitive metrics"""
    
    def test_calculate_cost_metrics(self):
        """Test cost-sensitive metrics calculation"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        calc = MetricsCalculator()
        cost_metrics = calc.calculate_cost_sensitive_metrics(
            y_true, y_pred,
            fp_cost=1.0,
            fn_cost=10.0
        )
        
        assert 'total_cost' in cost_metrics
        assert 'average_cost' in cost_metrics
        assert 'false_positives' in cost_metrics
        assert 'false_negatives' in cost_metrics
    
    def test_cost_metrics_values(self):
        """Test cost metrics have correct values"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])  # 1 FP, 1 FN
        
        calc = MetricsCalculator()
        cost_metrics = calc.calculate_cost_sensitive_metrics(
            y_true, y_pred,
            fp_cost=2.0,
            fn_cost=5.0
        )
        
        assert cost_metrics['false_positives'] == 1
        assert cost_metrics['false_negatives'] == 1
        assert cost_metrics['fp_cost_total'] == 2.0
        assert cost_metrics['fn_cost_total'] == 5.0
        assert cost_metrics['total_cost'] == 7.0


@pytest.mark.unit
class TestMetricsComparison:
    """Tests for model comparison"""
    
    def test_create_comparison_table(self):
        """Test creating metrics comparison table"""
        results = {
            'Model1': {'accuracy': 0.85, 'precision': 0.80},
            'Model2': {'accuracy': 0.90, 'precision': 0.85}
        }
        
        calc = MetricsCalculator()
        df = calc.create_metrics_comparison_table(results)
        
        assert isinstance(df, pd.DataFrame)
        assert 'Model1' in df.index
        assert 'Model2' in df.index
        assert 'accuracy' in df.columns


@pytest.mark.integration
class TestEvaluateModel:
    """Integration tests for model evaluation"""
    
    def test_evaluate_model_basic(self):
        """Test basic model evaluation"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 10)
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0] * 10)
        
        results = evaluate_model(
            y_true, y_pred,
            model_name="Test Model",
            plot=False
        )
        
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'classification_report' in results
    
    def test_evaluate_model_with_proba(self):
        """Test model evaluation with probabilities"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 10)
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0] * 10)
        y_proba = np.random.rand(80, 2)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        results = evaluate_model(
            y_true, y_pred, y_proba,
            model_name="Test Model",
            plot=False
        )
        
        assert 'roc_auc' in results['metrics']
    
    def test_evaluate_model_with_features(self):
        """Test model evaluation with feature importance"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0] * 10)
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0] * 10)
        feature_names = ['LOC', 'WMC', 'CBO', 'RFC', 'LCOM5']
        feature_importance = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
        
        results = evaluate_model(
            y_true, y_pred,
            feature_names=feature_names,
            feature_importance=feature_importance,
            plot=False
        )
        
        assert results is not None


@pytest.mark.unit
class TestMetricsValidation:
    """Tests for metrics validation"""
    
    def test_metrics_bounds(self):
        """Test that metrics are within valid bounds"""
        y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        for metric_name, value in metrics.items():
            if 'corrcoef' not in metric_name:  # MCC can be negative
                assert 0 <= value <= 1, f"{metric_name} out of bounds: {value}"
    
    def test_metrics_with_all_positive(self):
        """Test metrics when all predictions are positive"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        assert metrics['recall'] == 1.0  # All true positives caught
        assert metrics['precision'] < 1.0  # But also false positives
    
    def test_metrics_with_all_negative(self):
        """Test metrics when all predictions are negative"""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        assert metrics['recall'] == 0.0  # No true positives caught


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases"""
    
    def test_empty_predictions(self):
        """Test with empty predictions"""
        y_true = np.array([])
        y_pred = np.array([])
        
        calc = MetricsCalculator()
        # Should handle gracefully or raise informative error
        with pytest.raises(ValueError):
            calc.calculate_all_metrics(y_true, y_pred)
    
    def test_single_class_predictions(self):
        """Test with single class in predictions"""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0])  # All class 0
        
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_true, y_pred)
        
        # Should handle zero division
        assert 'precision' in metrics
        assert 'recall' in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
