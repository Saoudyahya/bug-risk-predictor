"""
Data preprocessing utilities for bug prediction
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN


class DataPreprocessor:
    """
    Comprehensive data preprocessing for bug prediction
    """
    
    def __init__(self):
        """Initialize the preprocessor"""
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
    
    def handle_missing_values(
        self,
        df: pd.DataFrame,
        strategy: str = 'median',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in the dataset
        
        Args:
            df: Input DataFrame
            strategy: Strategy for filling missing values ('mean', 'median', 'mode', 'drop', 'constant')
            fill_value: Value to use when strategy is 'constant'
            
        Returns:
            DataFrame with handled missing values
        """
        print(f"\nHandling missing values (strategy: {strategy})...")
        
        if strategy == 'drop':
            df_clean = df.dropna()
            print(f"Dropped {len(df) - len(df_clean)} rows with missing values")
            return df_clean
        
        elif strategy == 'mean':
            return df.fillna(df.mean())
        
        elif strategy == 'median':
            return df.fillna(df.median())
        
        elif strategy == 'mode':
            return df.fillna(df.mode().iloc[0])
        
        elif strategy == 'constant':
            if fill_value is None:
                fill_value = 0
            return df.fillna(fill_value)
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def remove_outliers(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers from the dataset
        
        Args:
            X: Feature DataFrame
            y: Target Series
            method: Method for outlier detection ('iqr', 'zscore')
            threshold: Threshold for outlier detection
            
        Returns:
            Tuple of (X_clean, y_clean)
        """
        print(f"\nRemoving outliers (method: {method})...")
        
        if method == 'iqr':
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Filter outliers
            mask = ~((X < lower_bound) | (X > upper_bound)).any(axis=1)
            
        elif method == 'zscore':
            z_scores = np.abs((X - X.mean()) / X.std())
            mask = (z_scores < threshold).all(axis=1)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Removed {len(X) - len(X_clean)} outliers")
        
        return X_clean, y_clean
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_test: Optional[pd.DataFrame] = None,
        method: str = 'standard'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Scale features using various methods
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            method: Scaling method ('standard', 'minmax', 'robust')
            
        Returns:
            Tuple of (X_train_scaled, X_test_scaled)
        """
        print(f"\nScaling features (method: {method})...")
        
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        elif method == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def select_features(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        k: int = 10,
        method: str = 'f_classif'
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], List[str]]:
        """
        Select top k features
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            k: Number of features to select
            method: Feature selection method ('f_classif', 'mutual_info')
            
        Returns:
            Tuple of (X_train_selected, X_test_selected, selected_feature_names)
        """
        print(f"\nSelecting top {k} features (method: {method})...")
        
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        # Get selected feature names
        feature_mask = self.feature_selector.get_support()
        self.selected_features = X_train.columns[feature_mask].tolist()
        
        print(f"Selected features: {self.selected_features}")
        
        X_test_selected = None
        if X_test is not None:
            X_test_selected = self.feature_selector.transform(X_test)
        
        return (
            pd.DataFrame(X_train_selected, columns=self.selected_features),
            pd.DataFrame(X_test_selected, columns=self.selected_features) if X_test_selected is not None else None,
            self.selected_features
        )
    
    def balance_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'smote',
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance imbalanced dataset
        
        Args:
            X: Features
            y: Labels
            method: Balancing method ('smote', 'adasyn', 'undersample', 'smoteenn')
            random_state: Random state for reproducibility
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        print(f"\nBalancing dataset (method: {method})...")
        print(f"Original class distribution: {np.bincount(y)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=random_state)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=random_state)
        elif method == 'smoteenn':
            sampler = SMOTEENN(random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_balanced, y_balanced = sampler.fit_resample(X, y)
        
        print(f"Balanced class distribution: {np.bincount(y_balanced)}")
        
        return X_balanced, y_balanced
    
    def create_polynomial_features(
        self,
        X: pd.DataFrame,
        degree: int = 2,
        interaction_only: bool = False
    ) -> pd.DataFrame:
        """
        Create polynomial features
        
        Args:
            X: Input features
            degree: Degree of polynomial features
            interaction_only: If True, only interaction features are produced
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        print(f"\nCreating polynomial features (degree={degree})...")
        
        poly = PolynomialFeatures(
            degree=degree,
            interaction_only=interaction_only,
            include_bias=False
        )
        
        X_poly = poly.fit_transform(X)
        feature_names = poly.get_feature_names_out(X.columns)
        
        print(f"Created {X_poly.shape[1]} features from {X.shape[1]} original features")
        
        return pd.DataFrame(X_poly, columns=feature_names, index=X.index)
    
    def preprocess_pipeline(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: Optional[pd.DataFrame] = None,
        y_test: Optional[pd.Series] = None,
        handle_missing: bool = True,
        remove_outliers: bool = False,
        scale: bool = True,
        select_features: bool = False,
        balance: bool = True,
        k_features: int = 15,
        balance_method: str = 'smote'
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Complete preprocessing pipeline
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            handle_missing: Whether to handle missing values
            remove_outliers: Whether to remove outliers
            scale: Whether to scale features
            select_features: Whether to select features
            balance: Whether to balance the dataset
            k_features: Number of features to select
            balance_method: Method for balancing
            
        Returns:
            Tuple of (X_train_processed, y_train_processed, X_test_processed, y_test_processed)
        """
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE")
        print("="*50)
        
        # Handle missing values
        if handle_missing:
            X_train = self.handle_missing_values(X_train, strategy='median')
            if X_test is not None:
                X_test = self.handle_missing_values(X_test, strategy='median')
        
        # Remove outliers (only from training set)
        if remove_outliers:
            X_train, y_train = self.remove_outliers(X_train, y_train)
        
        # Feature selection
        if select_features:
            X_train, X_test, selected_features = self.select_features(
                X_train, y_train, X_test, k=k_features
            )
        
        # Scale features
        if scale:
            X_train_scaled, X_test_scaled = self.scale_features(
                X_train, X_test, method='standard'
            )
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values if X_test is not None else None
        
        # Balance dataset (only training set)
        if balance:
            X_train_scaled, y_train = self.balance_dataset(
                X_train_scaled, y_train.values, method=balance_method
            )
        
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        print(f"Training set: {X_train_scaled.shape}")
        if X_test_scaled is not None:
            print(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_test_scaled, y_test


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        weights=[0.8, 0.2],  # Imbalanced
        random_state=42
    )
    
    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    X_train_proc, y_train_proc, X_test_proc, y_test_proc = preprocessor.preprocess_pipeline(
        X_train, y_train, X_test, y_test,
        select_features=True,
        k_features=10,
        balance=True
    )
    
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train_proc.shape}")
    print(f"y_train: {y_train_proc.shape}")
    print(f"X_test: {X_test_proc.shape}")
