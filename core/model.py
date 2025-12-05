"""
Base model class for bug prediction models
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import joblib
import numpy as np
from pathlib import Path


class BaseModel(ABC):
    """
    Abstract base class for all bug prediction models
    """
    
    def __init__(self, name: str = "BaseModel"):
        """
        Initialize the model
        
        Args:
            name: Name of the model
        """
        self.name = name
        self.model = None
        self.is_trained = False
        self.feature_names = None
        self.feature_importance = None
    
    @abstractmethod
    def build(self, input_dim: int, **kwargs) -> None:
        """
        Build the model architecture
        
        Args:
            input_dim: Number of input features
            **kwargs: Additional model-specific parameters
        """
        pass
    
    @abstractmethod
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels
        """
        pass
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, return one-hot predictions
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance scores
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.feature_importance is not None and self.feature_names is not None:
            return dict(zip(self.feature_names, self.feature_importance))
        return None
    
    def save(self, path: str) -> None:
        """
        Save the model to disk
        
        Args:
            path: Path to save the model
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'name': self.name,
            'model': self.model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, path: str) -> None:
        """
        Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        model_data = joblib.load(path)
        
        self.name = model_data['name']
        self.model = model_data['model']
        self.is_trained = model_data['is_trained']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data.get('feature_importance')
        
        print(f"Model loaded from {path}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', trained={self.is_trained})"


class EnsembleModel(BaseModel):
    """
    Ensemble model combining multiple base models
    """
    
    def __init__(self, models: list = None, name: str = "EnsembleModel"):
        """
        Initialize the ensemble model
        
        Args:
            models: List of base models to ensemble
            name: Name of the ensemble model
        """
        super().__init__(name)
        self.models = models or []
        self.weights = None
    
    def build(self, input_dim: int, **kwargs) -> None:
        """Build each model in the ensemble"""
        for model in self.models:
            model.build(input_dim, **kwargs)
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}: {model.name}")
            history = model.train(X_train, y_train, X_val, y_val, **kwargs)
            histories.append(history)
        
        self.is_trained = True
        return {'model_histories': histories}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using voting"""
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Majority voting
        predictions = np.array(predictions)
        final_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=predictions
        )
        
        return final_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted average"""
        probas = []
        
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Average probabilities
        avg_proba = np.mean(probas, axis=0)
        return avg_proba
