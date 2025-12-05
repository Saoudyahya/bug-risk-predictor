"""
Neural Network model for bug prediction using TensorFlow/Keras
"""

import numpy as np
from typing import Dict, Any, Optional, List
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from models.model import BaseModel


class NeuralNetworkModel(BaseModel):
    """
    Deep Neural Network for bug prediction
    """
    
    def __init__(
        self,
        name: str = "NeuralNetwork",
        hidden_layers: List[int] = [64, 32, 16],
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001
    ):
        """
        Initialize the Neural Network model
        
        Args:
            name: Model name
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        super().__init__(name)
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.scaler = StandardScaler()
        self.history = None
    
    def build(self, input_dim: int, **kwargs) -> None:
        """
        Build the neural network architecture
        
        Args:
            input_dim: Number of input features
            **kwargs: Additional parameters
        """
        # Input layer
        inputs = layers.Input(shape=(input_dim,))
        x = inputs
        
        # Hidden layers
        for i, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(0.01),
                name=f'hidden_{i+1}'
            )(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        self.model = keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"\nBuilt {self.name} model:")
        self.model.summary()
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the neural network
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            validation_data = (X_val_scaled, y_val)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        print(f"\nTraining {self.name}...")
        self.history = self.model.fit(
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose,
            **kwargs
        )
        
        self.is_trained = True
        
        # Calculate feature importance using gradients
        self._calculate_feature_importance(X_train_scaled)
        
        # Return history as dictionary
        return {k: [float(v) for v in vals] for k, vals in self.history.history.items()}
    
    def _calculate_feature_importance(self, X: np.ndarray) -> None:
        """
        Calculate feature importance using gradient-based method
        
        Args:
            X: Input features
        """
        try:
            # Take a sample for computation efficiency
            sample_size = min(1000, len(X))
            X_sample = X[:sample_size]
            
            # Convert to tensor
            X_tensor = tf.constant(X_sample, dtype=tf.float32)
            
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor, training=False)
            
            # Get gradients
            gradients = tape.gradient(predictions, X_tensor)
            
            # Calculate importance as mean absolute gradient
            self.feature_importance = np.mean(np.abs(gradients.numpy()), axis=0)
            
        except Exception as e:
            print(f"Warning: Could not calculate feature importance: {e}")
            self.feature_importance = None
    
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
        predictions = self.model.predict(X_scaled, verbose=0)
        return (predictions >= 0.5).astype(int).flatten()
    
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
        pos_proba = self.model.predict(X_scaled, verbose=0).flatten()
        
        # Return probabilities for both classes
        return np.column_stack([1 - pos_proba, pos_proba])
    
    def plot_history(self):
        """Plot training history"""
        try:
            import matplotlib.pyplot as plt
            
            if self.history is None:
                print("No training history available")
                return
            
            history_dict = self.history.history
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{self.name} Training History', fontsize=16)
            
            # Plot loss
            axes[0, 0].plot(history_dict['loss'], label='Train Loss')
            if 'val_loss' in history_dict:
                axes[0, 0].plot(history_dict['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # Plot accuracy
            axes[0, 1].plot(history_dict['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history_dict:
                axes[0, 1].plot(history_dict['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # Plot precision
            if 'precision' in history_dict:
                axes[1, 0].plot(history_dict['precision'], label='Train Precision')
                if 'val_precision' in history_dict:
                    axes[1, 0].plot(history_dict['val_precision'], label='Val Precision')
                axes[1, 0].set_title('Precision')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Precision')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Plot recall
            if 'recall' in history_dict:
                axes[1, 1].plot(history_dict['recall'], label='Train Recall')
                if 'val_recall' in history_dict:
                    axes[1, 1].plot(history_dict['val_recall'], label='Val Recall')
                axes[1, 1].set_title('Recall')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Recall')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


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
    model = NeuralNetworkModel(hidden_layers=[32, 16, 8])
    model.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    model.build(input_dim=X.shape[1])
    
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        batch_size=32
    )
    
    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    # Get probabilities
    y_proba = model.predict_proba(X_test)
    print(f"\nSample probabilities:\n{y_proba[:5]}")
