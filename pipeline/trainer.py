"""
Training pipeline for bug prediction models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

from core.dataset import BugDataset
from core.logistic_regression import LogisticRegressionModel
from core.neural_network import NeuralNetworkModel
from utils.preprocessing import DataPreprocessor
from utils.metrics import MetricsCalculator


class ModelTrainer:
    """
    Training pipeline for bug prediction models
    """
    
    def __init__(
        self,
        data_path: str = "data/",
        model_save_path: str = "models/"
    ):
        """
        Initialize the trainer
        
        Args:
            data_path: Path to data directory
            model_save_path: Path to save trained models
        """
        self.data_path = Path(data_path)
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        self.dataset = BugDataset(data_path)
        self.preprocessor = DataPreprocessor()
        self.metrics_calc = MetricsCalculator()
        
        self.models = {}
        self.results = {}
    
    def load_and_prepare_data(
        self,
        level: str = "class",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> None:
        """
        Load and prepare the dataset
        
        Args:
            level: Dataset level ('class' or 'file')
            test_size: Test set size
            random_state: Random state for reproducibility
        """
        print("\n" + "="*60)
        print("LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load dataset
        df = self.dataset.load_data(level=level)
        self.dataset.print_dataset_info(df)
        
        # Prepare features
        X, y = self.dataset.prepare_features(df)
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        self.feature_names = list(X.columns)
        
        print(f"\nTrain set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Train bug rate: {self.y_train.mean():.2%}")
        print(f"Test bug rate: {self.y_test.mean():.2%}")
    
    def preprocess_data(
        self,
        scale: bool = True,
        balance: bool = True,
        select_features: bool = False,
        k_features: int = 15
    ) -> None:
        """
        Preprocess the data
        
        Args:
            scale: Whether to scale features
            balance: Whether to balance dataset
            select_features: Whether to perform feature selection
            k_features: Number of features to select
        """
        X_train_proc, y_train_proc, X_test_proc, y_test_proc = self.preprocessor.preprocess_pipeline(
            self.X_train.copy(),
            self.y_train.copy(),
            self.X_test.copy(),
            self.y_test.copy(),
            handle_missing=True,
            remove_outliers=False,
            scale=scale,
            select_features=select_features,
            balance=balance,
            k_features=k_features,
            balance_method='smote'
        )
        
        self.X_train_proc = X_train_proc
        self.y_train_proc = y_train_proc
        self.X_test_proc = X_test_proc
        self.y_test_proc = y_test_proc
        
        # Update feature names if selection was performed
        if select_features and self.preprocessor.selected_features:
            self.feature_names = self.preprocessor.selected_features
    
    def train_logistic_regression(self, **kwargs) -> LogisticRegressionModel:
        """Train Logistic Regression model"""
        print("\n" + "="*60)
        print("TRAINING: Logistic Regression")
        print("="*60)
        
        model = LogisticRegressionModel(name="LogisticRegression")
        model.feature_names = self.feature_names
        model.build(input_dim=self.X_train_proc.shape[1])
        
        history = model.train(
            self.X_train_proc,
            self.y_train_proc,
            self.X_test_proc,
            self.y_test_proc
        )
        
        self.models['logistic_regression'] = model
        return model
    
    def train_random_forest(self, n_estimators: int = 100, **kwargs) -> RandomForestClassifier:
        """Train Random Forest model"""
        print("\n" + "="*60)
        print("TRAINING: Random Forest")
        print("="*60)
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(self.X_train_proc, self.y_train_proc)
        
        # Store feature importance
        self.models['random_forest'] = {
            'model': model,
            'feature_names': self.feature_names,
            'feature_importance': model.feature_importances_
        }
        
        print("Training complete!")
        return model
    
    def train_neural_network(
        self,
        hidden_layers: List[int] = [64, 32, 16],
        epochs: int = 100,
        **kwargs
    ) -> NeuralNetworkModel:
        """Train Neural Network model"""
        print("\n" + "="*60)
        print("TRAINING: Neural Network")
        print("="*60)
        
        model = NeuralNetworkModel(
            name="NeuralNetwork",
            hidden_layers=hidden_layers
        )
        model.feature_names = self.feature_names
        model.build(input_dim=self.X_train_proc.shape[1])
        
        history = model.train(
            self.X_train_proc,
            self.y_train_proc,
            self.X_test_proc,
            self.y_test_proc,
            epochs=epochs,
            batch_size=32
        )
        
        self.models['neural_network'] = model
        return model
    
    def evaluate_model(self, model_name: str, model) -> Dict:
        """Evaluate a trained model"""
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_name}")
        print(f"{'='*60}")
        
        # Handle different model types
        if isinstance(model, dict):
            # Random Forest or other sklearn models
            y_pred = model['model'].predict(self.X_test_proc)
            y_proba = model['model'].predict_proba(self.X_test_proc)
            feature_importance = model.get('feature_importance')
        else:
            # Custom models (LogisticRegression, NeuralNetwork)
            y_pred = model.predict(self.X_test_proc)
            y_proba = model.predict_proba(self.X_test_proc)
            feature_importance = model.feature_importance
        
        # Calculate metrics
        metrics = self.metrics_calc.calculate_all_metrics(
            self.y_test_proc,
            y_pred,
            y_proba
        )
        
        self.metrics_calc.print_metrics(metrics)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_proba
        }
        
        return metrics
    
    def save_model(self, model_name: str, filename: Optional[str] = None) -> None:
        """Save a trained model"""
        if model_name not in self.models:
            print(f"Model {model_name} not found!")
            return
        
        if filename is None:
            filename = f"{model_name}_model.pkl"
        
        save_path = self.model_save_path / filename
        
        model = self.models[model_name]
        
        if isinstance(model, dict):
            # For sklearn models, save the dictionary
            joblib.dump(model, save_path)
        else:
            # For custom models, use their save method
            model.save(save_path)
        
        print(f"Model saved to: {save_path}")
    
    def train_all_models(
        self,
        epochs: int = 100,
        save_models: bool = True
    ) -> Dict[str, Dict]:
        """Train all available models"""
        results = {}
        
        # Logistic Regression
        lr_model = self.train_logistic_regression()
        results['Logistic Regression'] = self.evaluate_model('logistic_regression', lr_model)
        if save_models:
            self.save_model('logistic_regression')
        
        # Random Forest
        rf_model = self.train_random_forest()
        results['Random Forest'] = self.evaluate_model('random_forest', self.models['random_forest'])
        if save_models:
            self.save_model('random_forest')
        
        # Neural Network
        nn_model = self.train_neural_network(epochs=epochs)
        results['Neural Network'] = self.evaluate_model('neural_network', nn_model)
        if save_models:
            self.save_model('neural_network')
        
        # Print comparison
        self.print_results_comparison(results)
        
        return results
    
    def print_results_comparison(self, results: Dict[str, Dict]) -> None:
        """Print comparison of all models"""
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        df = pd.DataFrame(results).T
        print(df.round(4))
        
        # Find best model for each metric
        print("\nBest Models:")
        for metric in df.columns:
            best_model = df[metric].idxmax()
            best_score = df[metric].max()
            print(f"  {metric}: {best_model} ({best_score:.4f})")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train bug prediction models')
    parser.add_argument('--model', type=str, default='all',
                       choices=['all', 'lr', 'rf', 'nn'],
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs for neural network')
    parser.add_argument('--data-path', type=str, default='data/',
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str, default='models/',
                       help='Path to save models')
    parser.add_argument('--no-balance', action='store_true',
                       help='Do not balance dataset')
    parser.add_argument('--select-features', action='store_true',
                       help='Perform feature selection')
    parser.add_argument('--k-features', type=int, default=15,
                       help='Number of features to select')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        data_path=args.data_path,
        model_save_path=args.model_path
    )
    
    # Load and prepare data
    trainer.load_and_prepare_data()
    
    # Preprocess
    trainer.preprocess_data(
        balance=not args.no_balance,
        select_features=args.select_features,
        k_features=args.k_features
    )
    
    # Train models
    if args.model == 'all':
        trainer.train_all_models(epochs=args.epochs)
    elif args.model == 'lr':
        model = trainer.train_logistic_regression()
        trainer.evaluate_model('logistic_regression', model)
        trainer.save_model('logistic_regression')
    elif args.model == 'rf':
        model = trainer.train_random_forest()
        trainer.evaluate_model('random_forest', trainer.models['random_forest'])
        trainer.save_model('random_forest')
    elif args.model == 'nn':
        model = trainer.train_neural_network(epochs=args.epochs)
        trainer.evaluate_model('neural_network', model)
        trainer.save_model('neural_network')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
