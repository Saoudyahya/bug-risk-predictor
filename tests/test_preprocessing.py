"""
Tests for preprocessing module
"""

import pytest
import numpy as np
import pandas as pd
from utils.preprocessing import DataPreprocessor


@pytest.mark.preprocessing
@pytest.mark.unit
class TestMissingValueHandling:
    """Tests for missing value handling"""
    
    def test_handle_missing_median(self):
        """Test handling missing values with median strategy"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50]
        })
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='median')
        
        assert df_clean.isnull().sum().sum() == 0
        assert df_clean['A'].iloc[2] == df['A'].median()
    
    def test_handle_missing_mean(self):
        """Test handling missing values with mean strategy"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50]
        })
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='mean')
        
        assert df_clean.isnull().sum().sum() == 0
    
    def test_handle_missing_drop(self):
        """Test dropping rows with missing values"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, np.nan, 50]
        })
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='drop')
        
        assert df_clean.isnull().sum().sum() == 0
        assert len(df_clean) < len(df)
    
    def test_handle_missing_constant(self):
        """Test filling missing values with constant"""
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5]
        })
        
        preprocessor = DataPreprocessor()
        df_clean = preprocessor.handle_missing_values(df, strategy='constant', fill_value=999)
        
        assert df_clean.isnull().sum().sum() == 0
        assert df_clean['A'].iloc[2] == 999


@pytest.mark.preprocessing
@pytest.mark.unit
class TestOutlierRemoval:
    """Tests for outlier removal"""
    
    def test_remove_outliers_iqr(self):
        """Test outlier removal using IQR method"""
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100],  # 100 is outlier
            'B': [10, 20, 30, 40, 50, 60]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        preprocessor = DataPreprocessor()
        X_clean, y_clean = preprocessor.remove_outliers(X, y, method='iqr')
        
        assert len(X_clean) < len(X)
        assert 100 not in X_clean['A'].values
    
    def test_remove_outliers_zscore(self):
        """Test outlier removal using Z-score method"""
        X = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100]
        })
        y = pd.Series([0, 0, 0, 1, 1, 1])
        
        preprocessor = DataPreprocessor()
        X_clean, y_clean = preprocessor.remove_outliers(X, y, method='zscore', threshold=3)
        
        assert len(X_clean) < len(X)


@pytest.mark.preprocessing
@pytest.mark.unit
class TestFeatureScaling:
    """Tests for feature scaling"""
    
    def test_standard_scaling(self):
        """Test standard scaling"""
        X_train = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50]
        })
        X_test = pd.DataFrame({
            'A': [6, 7],
            'B': [60, 70]
        })
        
        preprocessor = DataPreprocessor()
        X_train_scaled, X_test_scaled = preprocessor.scale_features(
            X_train, X_test, method='standard'
        )
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        # Check that mean is close to 0 and std is close to 1
        assert abs(X_train_scaled.mean()) < 1e-10
        assert abs(X_train_scaled.std() - 1.0) < 0.1
    
    def test_minmax_scaling(self):
        """Test min-max scaling"""
        X_train = pd.DataFrame({
            'A': [1, 2, 3, 4, 5]
        })
        
        preprocessor = DataPreprocessor()
        X_train_scaled, _ = preprocessor.scale_features(
            X_train, method='minmax'
        )
        
        # Check that values are between 0 and 1
        assert X_train_scaled.min() >= 0
        assert X_train_scaled.max() <= 1
    
    def test_robust_scaling(self):
        """Test robust scaling"""
        X_train = pd.DataFrame({
            'A': [1, 2, 3, 4, 5, 100]  # Outlier present
        })
        
        preprocessor = DataPreprocessor()
        X_train_scaled, _ = preprocessor.scale_features(
            X_train, method='robust'
        )
        
        assert X_train_scaled.shape == X_train.shape


@pytest.mark.preprocessing
@pytest.mark.unit
class TestFeatureSelection:
    """Tests for feature selection"""
    
    def test_select_features_f_classif(self):
        """Test feature selection with f_classif"""
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        preprocessor = DataPreprocessor()
        X_train_selected, _, selected_features = preprocessor.select_features(
            X_train, y_train, k=10, method='f_classif'
        )
        
        assert X_train_selected.shape[1] == 10
        assert len(selected_features) == 10
        assert preprocessor.selected_features == selected_features
    
    def test_select_features_mutual_info(self):
        """Test feature selection with mutual information"""
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        preprocessor = DataPreprocessor()
        X_train_selected, _, selected_features = preprocessor.select_features(
            X_train, y_train, k=5, method='mutual_info'
        )
        
        assert X_train_selected.shape[1] == 5
        assert len(selected_features) == 5
    
    def test_select_features_with_test_set(self):
        """Test feature selection with test set"""
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        X_test = pd.DataFrame(np.random.rand(20, 20))
        X_test.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        preprocessor = DataPreprocessor()
        X_train_selected, X_test_selected, _ = preprocessor.select_features(
            X_train, y_train, X_test, k=10
        )
        
        assert X_train_selected.shape[1] == 10
        assert X_test_selected.shape[1] == 10


@pytest.mark.preprocessing
@pytest.mark.unit
class TestDatasetBalancing:
    """Tests for dataset balancing"""
    
    def test_balance_smote(self):
        """Test SMOTE balancing"""
        X = np.random.rand(100, 10)
        y = np.array([0] * 80 + [1] * 20)  # Imbalanced
        
        preprocessor = DataPreprocessor()
        X_balanced, y_balanced = preprocessor.balance_dataset(
            X, y, method='smote'
        )
        
        # Check that classes are balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        assert len(unique) == 2
        assert abs(counts[0] - counts[1]) < 5  # Nearly balanced
    
    def test_balance_undersample(self):
        """Test undersampling"""
        X = np.random.rand(100, 10)
        y = np.array([0] * 80 + [1] * 20)
        
        preprocessor = DataPreprocessor()
        X_balanced, y_balanced = preprocessor.balance_dataset(
            X, y, method='undersample'
        )
        
        # Should have fewer samples
        assert len(X_balanced) < len(X)
        
        # Should be balanced
        unique, counts = np.unique(y_balanced, return_counts=True)
        assert counts[0] == counts[1]
    
    def test_balance_adasyn(self):
        """Test ADASYN balancing"""
        X = np.random.rand(100, 10)
        y = np.array([0] * 80 + [1] * 20)
        
        preprocessor = DataPreprocessor()
        X_balanced, y_balanced = preprocessor.balance_dataset(
            X, y, method='adasyn'
        )
        
        # Check that minority class was oversampled
        assert len(X_balanced) > len(X)


@pytest.mark.preprocessing
@pytest.mark.integration
class TestPreprocessingPipeline:
    """Integration tests for preprocessing pipeline"""
    
    def test_complete_pipeline(self):
        """Test complete preprocessing pipeline"""
        # Create sample data
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        X_test = pd.DataFrame(np.random.rand(20, 20))
        X_test.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.array([0] * 80 + [1] * 20))
        y_test = pd.Series(np.array([0] * 16 + [1] * 4))
        
        # Add some missing values
        X_train.iloc[0, 0] = np.nan
        X_test.iloc[0, 0] = np.nan
        
        preprocessor = DataPreprocessor()
        X_train_proc, y_train_proc, X_test_proc, y_test_proc = preprocessor.preprocess_pipeline(
            X_train, y_train, X_test, y_test,
            handle_missing=True,
            scale=True,
            select_features=True,
            balance=True,
            k_features=10
        )
        
        # Check shapes
        assert X_train_proc.shape[1] == 10  # Feature selection
        assert X_test_proc.shape[1] == 10
        
        # Check no missing values
        assert not np.isnan(X_train_proc).any()
        assert not np.isnan(X_test_proc).any()
        
        # Check balancing
        unique, counts = np.unique(y_train_proc, return_counts=True)
        assert abs(counts[0] - counts[1]) < 10
    
    def test_pipeline_without_test_set(self):
        """Test pipeline without test set"""
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.array([0] * 80 + [1] * 20))
        
        preprocessor = DataPreprocessor()
        X_train_proc, y_train_proc, X_test_proc, y_test_proc = preprocessor.preprocess_pipeline(
            X_train, y_train,
            handle_missing=True,
            scale=True,
            balance=True
        )
        
        assert X_test_proc is None
        assert y_test_proc is None
        assert X_train_proc.shape[0] > X_train.shape[0]  # Due to balancing
    
    def test_pipeline_minimal_processing(self):
        """Test pipeline with minimal processing"""
        X_train = pd.DataFrame(np.random.rand(100, 20))
        X_train.columns = [f'Feature_{i}' for i in range(20)]
        y_train = pd.Series(np.random.randint(0, 2, 100))
        
        preprocessor = DataPreprocessor()
        X_train_proc, y_train_proc, _, _ = preprocessor.preprocess_pipeline(
            X_train, y_train,
            handle_missing=False,
            scale=False,
            select_features=False,
            balance=False
        )
        
        # Should be mostly unchanged
        assert X_train_proc.shape == X_train.shape


@pytest.mark.preprocessing
@pytest.mark.unit
class TestPolynomialFeatures:
    """Tests for polynomial feature creation"""
    
    def test_create_polynomial_features(self):
        """Test polynomial feature creation"""
        X = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        preprocessor = DataPreprocessor()
        X_poly = preprocessor.create_polynomial_features(X, degree=2)
        
        assert X_poly.shape[1] > X.shape[1]
        assert X_poly.shape[0] == X.shape[0]
    
    def test_polynomial_interaction_only(self):
        """Test interaction-only polynomial features"""
        X = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        preprocessor = DataPreprocessor()
        X_poly = preprocessor.create_polynomial_features(
            X, degree=2, interaction_only=True
        )
        
        # Should have fewer features than full polynomial
        assert X_poly.shape[1] < 6  # Less than full degree-2 expansion


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
