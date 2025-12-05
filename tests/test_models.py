"""
Tests for model classes
"""

import pytest
import numpy as np
from pathlib import Path
from sklearn.datasets import make_classification


@pytest.mark.model
@pytest.mark.unit
class TestBaseModel:
    """Tests for BaseModel class"""
    
    def test_base_model_import(self):
        """Test importing BaseModel"""
        from models.model import BaseModel
        assert BaseModel is not None
    
    def test_base_model_is_abstract(self):
        """Test that BaseModel is abstract"""
        from models.model import BaseModel
        
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseModel()


@pytest.mark.model
@pytest.mark.unit
class TestLogisticRegressionModel:
    """Tests for Logistic Regression model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        from core.logistic_regression import LogisticRegressionModel
        
        model = LogisticRegressionModel()
        assert model.name == "LogisticRegression"
        assert model.is_trained is False
    
    def test_model_build(self):
        """Test building the model"""
        from core.logistic_regression import LogisticRegressionModel
        
        model = LogisticRegressionModel()
        model.build(input_dim=10)
        
        assert model.model is not None
    
    def test_model_train(self, train_test_data):
        """Test training the model"""
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel()
        model.build(input_dim=X_train.shape[1])
        history = model.train(X_train, y_train, X_test, y_test)
        
        assert model.is_trained is True
        assert 'train_accuracy' in history
        assert history['train_accuracy'] > 0
    
    def test_model_predict(self, train_test_data):
        """Test making predictions"""
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel()
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_model_predict_proba(self, train_test_data):
        """Test probability predictions"""
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel()
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_model_feature_importance(self, train_test_data):
        """Test feature importance"""
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = LogisticRegressionModel()
        model.feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train)
        
        top_features = model.get_top_features(n=5)
        
        assert len(top_features) == 5
        assert all(isinstance(v, float) for v in top_features.values())
    
    def test_model_save_load(self, train_test_data, model_save_path):
        """Test saving and loading model"""
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Train and save
        model = LogisticRegressionModel()
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train)
        
        save_path = model_save_path / "test_lr.pkl"
        model.save(str(save_path))
        
        # Load and test
        model2 = LogisticRegressionModel()
        model2.load(str(save_path))
        
        assert model2.is_trained is True
        predictions = model2.predict(X_test)
        assert len(predictions) == len(X_test)


@pytest.mark.model
@pytest.mark.slow
class TestNeuralNetworkModel:
    """Tests for Neural Network model"""
    
    def test_model_initialization(self):
        """Test model initialization"""
        from core.neural_network import NeuralNetworkModel
        
        model = NeuralNetworkModel()
        assert model.name == "NeuralNetwork"
        assert model.is_trained is False
    
    def test_model_build(self):
        """Test building the model"""
        from core.neural_network import NeuralNetworkModel
        
        model = NeuralNetworkModel(hidden_layers=[32, 16])
        model.build(input_dim=10)
        
        assert model.model is not None
    
    def test_model_train(self, train_test_data):
        """Test training the model"""
        from core.neural_network import NeuralNetworkModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = NeuralNetworkModel(hidden_layers=[16, 8])
        model.build(input_dim=X_train.shape[1])
        history = model.train(
            X_train, y_train, X_test, y_test,
            epochs=5,
            batch_size=32,
            verbose=0
        )
        
        assert model.is_trained is True
        assert 'loss' in history
        assert 'accuracy' in history
    
    def test_model_predict(self, train_test_data):
        """Test making predictions"""
        from core.neural_network import NeuralNetworkModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = NeuralNetworkModel(hidden_layers=[16])
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train, epochs=5, verbose=0)
        
        predictions = model.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})
    
    def test_model_predict_proba(self, train_test_data):
        """Test probability predictions"""
        from core.neural_network import NeuralNetworkModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = NeuralNetworkModel(hidden_layers=[16])
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train, epochs=5, verbose=0)
        
        probas = model.predict_proba(X_test)
        
        assert probas.shape == (len(X_test), 2)
        assert np.allclose(probas.sum(axis=1), 1.0, atol=0.01)
    
    def test_model_early_stopping(self, train_test_data):
        """Test early stopping callback"""
        from core.neural_network import NeuralNetworkModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        model = NeuralNetworkModel()
        model.build(input_dim=X_train.shape[1])
        history = model.train(
            X_train, y_train, X_test, y_test,
            epochs=100,
            verbose=0
        )
        
        # Should stop early
        actual_epochs = len(history['loss'])
        assert actual_epochs < 100


@pytest.mark.model
@pytest.mark.unit
class TestEnsembleModel:
    """Tests for Ensemble model"""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization"""
        from models.model import EnsembleModel
        from core.logistic_regression import LogisticRegressionModel
        
        models = [
            LogisticRegressionModel(name="LR1"),
            LogisticRegressionModel(name="LR2")
        ]
        
        ensemble = EnsembleModel(models=models)
        assert len(ensemble.models) == 2
    
    def test_ensemble_train(self, train_test_data):
        """Test training ensemble"""
        from models.model import EnsembleModel
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        models = [
            LogisticRegressionModel(name="LR1"),
            LogisticRegressionModel(name="LR2")
        ]
        
        ensemble = EnsembleModel(models=models)
        ensemble.build(input_dim=X_train.shape[1])
        ensemble.train(X_train, y_train)
        
        assert ensemble.is_trained is True
    
    def test_ensemble_predict(self, train_test_data):
        """Test ensemble predictions"""
        from models.model import EnsembleModel
        from core.logistic_regression import LogisticRegressionModel
        
        X_train, X_test, y_train, y_test = train_test_data
        
        models = [
            LogisticRegressionModel(name="LR1"),
            LogisticRegressionModel(name="LR2")
        ]
        
        ensemble = EnsembleModel(models=models)
        ensemble.build(input_dim=X_train.shape[1])
        ensemble.train(X_train, y_train)
        
        predictions = ensemble.predict(X_test)
        
        assert len(predictions) == len(X_test)
        assert set(predictions).issubset({0, 1})


@pytest.mark.model
@pytest.mark.integration
class TestModelWorkflow:
    """Integration tests for complete model workflow"""
    
    def test_logistic_regression_workflow(self, train_test_data, model_save_path):
        """Test complete LR workflow"""
        from core.logistic_regression import LogisticRegressionModel
        from utils.metrics import MetricsCalculator
        
        X_train, X_test, y_train, y_test = train_test_data
        
        # Create and train
        model = LogisticRegressionModel()
        model.feature_names = [f'F{i}' for i in range(X_train.shape[1])]
        model.build(input_dim=X_train.shape[1])
        model.train(X_train, y_train, X_test, y_test)
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Evaluate
        calc = MetricsCalculator()
        metrics = calc.calculate_all_metrics(y_test, y_pred, y_proba)
        
        assert metrics['accuracy'] > 0.5
        
        # Save
        save_path = model_save_path / "lr_workflow.pkl"
        model.save(str(save_path))
        
        assert save_path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
