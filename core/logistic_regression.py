"""
Logistic Regression model for bug prediction
"""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from models.model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model for binary bug prediction
    """
    
    def __init__(
        self, 
        name: str = "LogisticRegression",
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = 'balanced'
    ):
        """
        Initialize the Logistic Regression model
        
        Args:
            name: Model name
            C: Inverse regularization strength
            max_iter: Maximum iterations for optimization
            class_weight: Strategy for handling imbalanced classes
        """
        super().__init__(name)
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.scaler = StandardScaler()
    
    def build(self, input_dim: int, **kwargs) -> None:
        """
        Build the logistic regression model
        
        Args:
            input_dim: Number of input features
            **kwargs: Additional parameters for LogisticRegression
        """
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            random_state=42,
            **kwargs
        )
        print(f"Built {self.name} model with {input_dim} features")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the logistic regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (not used for LR)
            y_val: Validation labels (not used for LR)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training metrics
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train model
        print(f"Training {self.name}...")
        self.model.fit(X_train_scaled, y_train)
        
        # Calculate feature importance (absolute coefficients)
        self.feature_importance = np.abs(self.model.coef_[0])
        
        # Training accuracy
        train_acc = self.model.score(X_train_scaled, y_train)
        
        self.is_trained = True
        
        history = {
            'train_accuracy': train_acc,
            'coefficients': self.model.coef_[0],
            'intercept': self.model.intercept_[0]
        }
        
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_acc = self.model.score(X_val_scaled, y_val)
            history['val_accuracy'] = val_acc
        
        print(f"Training complete. Train accuracy: {train_acc:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_top_features(self, n: int = 10) -> Dict[str, float]:
        """
        Get top N most important features
        
        Args:
            n: Number of top features to return
            
        Returns:
            Dictionary of top features and their importance
        """
        if self.feature_importance is None:
            return {}
        
        if self.feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(self.feature_importance))]
        else:
            feature_names = self.feature_names
        
        # Sort by importance
        indices = np.argsort(self.feature_importance)[::-1][:n]
        
        top_features = {
            feature_names[i]: float(self.feature_importance[i])
            for i in indices
        }
        
        return top_features


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = LogisticRegressionModel()
    model.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    model.build(input_dim=X.shape[1])
    
    history = model.train(X_train, y_train, X_test, y_test)
    print(f"\nTraining History: {history}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Get top features
    top_features = model.get_top_features(n=5)
    print(f"\nTop 5 Features:")
    for feature, importance in top_features.items():
        print(f"  {feature}: {importance:.4f}")
