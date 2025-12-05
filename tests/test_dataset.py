"""
Tests for dataset module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from core.dataset import BugDataset


@pytest.mark.unit
class TestBugDataset:
    """Tests for BugDataset class"""
    
    def test_init(self, temp_dir):
        """Test dataset initialization"""
        dataset = BugDataset(str(temp_dir))
        
        assert dataset.data_path == temp_dir
        assert dataset.df_classes is None
        assert dataset.df_files is None
    
    def test_create_sample_dataset(self, temp_dir):
        """Test sample dataset creation"""
        dataset = BugDataset(str(temp_dir))
        df = dataset._create_sample_dataset()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'bug' in df.columns
        assert 'LOC' in df.columns
        assert 'WMC' in df.columns
    
    def test_load_data_creates_sample(self, temp_dir):
        """Test that load_data creates sample if no file exists"""
        dataset = BugDataset(str(temp_dir))
        df = dataset.load_data(level="class")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'bug' in df.columns
    
    def test_load_data_from_file(self, sample_csv_file):
        """Test loading data from existing file"""
        dataset = BugDataset(str(sample_csv_file.parent))
        df = dataset.load_data(sample_file=str(sample_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


@pytest.mark.unit
class TestMetricsColumns:
    """Tests for metrics column extraction"""
    
    def test_get_metrics_columns(self, sample_dataframe):
        """Test getting metrics columns"""
        dataset = BugDataset()
        metrics_cols = dataset.get_metrics_columns(sample_dataframe)
        
        assert 'Name' not in metrics_cols
        assert 'Path' not in metrics_cols
        assert 'bug' not in metrics_cols
        assert 'LOC' in metrics_cols
        assert 'WMC' in metrics_cols
    
    def test_metrics_columns_all_numeric(self, sample_dataframe):
        """Test that metrics columns are numeric"""
        dataset = BugDataset()
        metrics_cols = dataset.get_metrics_columns(sample_dataframe)
        
        for col in metrics_cols:
            assert pd.api.types.is_numeric_dtype(sample_dataframe[col])


@pytest.mark.unit
class TestFeaturePreparation:
    """Tests for feature preparation"""
    
    def test_prepare_features(self, sample_dataframe):
        """Test feature preparation"""
        dataset = BugDataset()
        X, y = dataset.prepare_features(sample_dataframe)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)
        assert len(X) == len(sample_dataframe)
    
    def test_prepare_features_no_missing_values(self, sample_dataframe):
        """Test that prepared features have no missing values"""
        dataset = BugDataset()
        X, y = dataset.prepare_features(sample_dataframe)
        
        assert X.isnull().sum().sum() == 0
    
    def test_prepare_features_binary_target(self, sample_dataframe):
        """Test that target is binary"""
        dataset = BugDataset()
        X, y = dataset.prepare_features(sample_dataframe)
        
        assert set(y.unique()).issubset({0, 1})
    
    def test_prepare_features_with_missing_values(self):
        """Test feature preparation with missing values"""
        df = pd.DataFrame({
            'LOC': [100, 200, np.nan, 400],
            'WMC': [10, np.nan, 30, 40],
            'bug': [0, 1, 0, 1]
        })
        
        dataset = BugDataset()
        X, y = dataset.prepare_features(df)
        
        assert X.isnull().sum().sum() == 0  # All NaNs filled
    
    def test_prepare_features_custom_metrics(self, sample_dataframe):
        """Test feature preparation with custom metrics"""
        dataset = BugDataset()
        custom_metrics = ['LOC', 'WMC', 'CBO']
        X, y = dataset.prepare_features(sample_dataframe, metrics=custom_metrics)
        
        assert list(X.columns) == custom_metrics


@pytest.mark.unit
class TestDatasetInfo:
    """Tests for dataset information"""
    
    def test_get_dataset_info(self, sample_dataframe):
        """Test getting dataset information"""
        dataset = BugDataset()
        info = dataset.get_dataset_info(sample_dataframe)
        
        assert 'total_samples' in info
        assert 'buggy_samples' in info
        assert 'clean_samples' in info
        assert 'bug_rate' in info
        assert 'n_features' in info
        assert 'feature_names' in info
    
    def test_dataset_info_values(self, sample_dataframe):
        """Test dataset info values are correct"""
        dataset = BugDataset()
        info = dataset.get_dataset_info(sample_dataframe)
        
        assert info['total_samples'] == len(sample_dataframe)
        assert info['buggy_samples'] + info['clean_samples'] == info['total_samples']
        assert 0 <= info['bug_rate'] <= 1
        assert info['n_features'] > 0
    
    def test_print_dataset_info(self, sample_dataframe, capsys):
        """Test printing dataset information"""
        dataset = BugDataset()
        dataset.print_dataset_info(sample_dataframe)
        
        captured = capsys.readouterr()
        assert "DATASET INFORMATION" in captured.out
        assert "Total Samples" in captured.out
        assert "Buggy Samples" in captured.out


@pytest.mark.unit
class TestSampleDataset:
    """Tests for sample dataset creation"""
    
    def test_sample_dataset_structure(self, temp_dir):
        """Test sample dataset has correct structure"""
        dataset = BugDataset(str(temp_dir))
        df = dataset._create_sample_dataset()
        
        required_columns = ['Name', 'Path', 'LOC', 'WMC', 'CBO', 'bug']
        for col in required_columns:
            assert col in df.columns
    
    def test_sample_dataset_size(self, temp_dir):
        """Test sample dataset has expected size"""
        dataset = BugDataset(str(temp_dir))
        df = dataset._create_sample_dataset()
        
        assert len(df) == 1000  # Default size
    
    def test_sample_dataset_saved(self, temp_dir):
        """Test sample dataset is saved to file"""
        dataset = BugDataset(str(temp_dir))
        df = dataset._create_sample_dataset()
        
        saved_file = temp_dir / "sample_bug_dataset.csv"
        assert saved_file.exists()
    
    def test_sample_dataset_bug_correlation(self, temp_dir):
        """Test that bug labels correlate with complexity"""
        dataset = BugDataset(str(temp_dir))
        df = dataset._create_sample_dataset()
        
        # Higher complexity should lead to more bugs
        high_complexity = df[df['WMC'] > 20]
        low_complexity = df[df['WMC'] <= 20]
        
        high_bug_rate = high_complexity['bug'].mean()
        low_bug_rate = low_complexity['bug'].mean()
        
        assert high_bug_rate >= low_bug_rate


@pytest.mark.unit
class TestImportantMetrics:
    """Tests for important metrics definition"""
    
    def test_important_metrics_defined(self):
        """Test that important metrics are defined"""
        assert len(BugDataset.IMPORTANT_METRICS) > 0
    
    def test_important_metrics_coverage(self):
        """Test that important metrics cover key categories"""
        metrics = BugDataset.IMPORTANT_METRICS
        
        # Check for size metrics
        size_metrics = [m for m in metrics if 'LOC' in m]
        assert len(size_metrics) > 0
        
        # Check for complexity metrics
        complexity_metrics = [m for m in metrics if any(x in m for x in ['WMC', 'McCabe'])]
        assert len(complexity_metrics) > 0
        
        # Check for coupling metrics
        coupling_metrics = [m for m in metrics if any(x in m for x in ['CBO', 'RFC'])]
        assert len(coupling_metrics) > 0


@pytest.mark.integration
class TestDatasetWorkflow:
    """Integration tests for dataset workflow"""
    
    def test_complete_workflow(self, temp_dir):
        """Test complete dataset workflow"""
        dataset = BugDataset(str(temp_dir))
        
        # Load data
        df = dataset.load_data(level="class")
        assert isinstance(df, pd.DataFrame)
        
        # Get info
        info = dataset.get_dataset_info(df)
        assert info['total_samples'] > 0
        
        # Prepare features
        X, y = dataset.prepare_features(df)
        assert len(X) == len(y)
        
        # Check no missing values
        assert X.isnull().sum().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
