"""
Model optimizer for hyperparameter tuning
"""

import numpy as np
from typing import Dict, Any, Callable
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import optuna


class ModelOptimizer:
    """
    Hyperparameter optimization for bug prediction models
    """
    
    def __init__(self, model, scoring: str = 'f1'):
        """
        Initialize the optimizer
        
        Args:
            model: Model to optimize
            scoring: Scoring metric ('f1', 'accuracy', 'precision', 'recall')
        """
        self.model = model
        self.scoring = scoring
        self.best_params = None
        self.best_score = None
    
    def grid_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_grid: Dict[str, list],
        cv: int = 5,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform grid search for hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and score
        """
        print(f"Starting grid search with {len(param_grid)} parameter combinations...")
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            scoring=self.scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': grid_search.cv_results_
        }
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best {self.scoring} score: {self.best_score:.4f}")
        
        return results
    
    def random_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        param_distributions: Dict[str, Any],
        n_iter: int = 100,
        cv: int = 5,
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform randomized search for hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training labels
            param_distributions: Dictionary of parameter distributions
            n_iter: Number of parameter settings to sample
            cv: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and score
        """
        print(f"Starting random search with {n_iter} iterations...")
        
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=self.scoring,
            cv=cv,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'cv_results': random_search.cv_results_
        }
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best {self.scoring} score: {self.best_score:.4f}")
        
        return results
    
    def bayesian_optimization(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        objective_func: Callable,
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna
        
        Args:
            X_train: Training features
            y_train: Training labels
            objective_func: Objective function for optimization
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with best parameters and score
        """
        print(f"Starting Bayesian optimization with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        # Create wrapper for objective function
        def objective(trial):
            return objective_func(trial, X_train, y_train, self.model)
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'study': study
        }
        
        print(f"\nBest parameters: {self.best_params}")
        print(f"Best {self.scoring} score: {self.best_score:.4f}")
        
        return results


# Example objective functions for different models

def logistic_regression_objective(trial, X_train, y_train, model):
    """Objective function for Logistic Regression"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    
    # Suggest parameters
    C = trial.suggest_loguniform('C', 1e-3, 1e3)
    max_iter = trial.suggest_int('max_iter', 100, 2000)
    
    # Create model
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight='balanced',
        random_state=42
    )
    
    # Cross-validation score
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    return scores.mean()


def random_forest_objective(trial, X_train, y_train, model):
    """Objective function for Random Forest"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    # Suggest parameters
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    # Create model
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation score
    scores = cross_val_score(
        clf, X_train, y_train,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    return scores.mean()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize model and optimizer
    model = LogisticRegression(random_state=42)
    optimizer = ModelOptimizer(model, scoring='f1')
    
    # Grid search
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'max_iter': [100, 500, 1000]
    }
    
    results = optimizer.grid_search(X_train, y_train, param_grid)
    print(f"\nGrid Search Results:")
    print(f"Best Params: {results['best_params']}")
    print(f"Best Score: {results['best_score']:.4f}")
