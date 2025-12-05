"""
Dataset module for loading and processing bug prediction datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict
import requests
from tqdm import tqdm
import zipfile
import os


class BugDataset:
    """
    Handler for the Public Unified Bug Dataset
    """
    
    DATASET_URL = "http://www.inf.u-szeged.hu/~ferenc/papers/UnifiedBugDataSet/bug-data-1.2.zip"
    
    # Important software metrics
    IMPORTANT_METRICS = [
        'LOC', 'LLOC', 'NOS', 'TLOC', 'TLLOC',  # Size metrics
        'WMC', 'McCabe', 'NL', 'NLE',  # Complexity metrics
        'CBO', 'CBOI', 'RFC', 'NII', 'NOI',  # Coupling metrics
        'LCOM5', 'TCC', 'LCC',  # Cohesion metrics
        'DIT', 'NOC', 'NOA', 'NOM', 'NOP',  # Inheritance metrics
    ]
    
    def __init__(self, data_path: str = "../data"):
        """
        Initialize the dataset handler

        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            self.data_path.mkdir(parents=True, exist_ok=True)

        self.df_classes = None
        self.df_files = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def download_dataset(self, force: bool = False) -> None:
        """
        Download the Public Unified Bug Dataset

        Args:
            force: Force re-download even if file exists
        """
        zip_path = self.data_path / "bug-data-1.2.zip"

        if zip_path.exists() and not force:
            print(f"Dataset already downloaded at {zip_path}")
            return

        print(f"Downloading dataset from {self.DATASET_URL}...")

        try:
            response = requests.get(self.DATASET_URL, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            print("Download complete. Extracting...")
            self._extract_dataset(zip_path)

        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("\nPlease download manually from:")
            print(self.DATASET_URL)
            print(f"And place it in {self.data_path}")

    def _extract_dataset(self, zip_path: Path) -> None:
        """Extract the downloaded dataset"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
            print(f"Dataset extracted to {self.data_path}")
        except Exception as e:
            print(f"Error extracting dataset: {e}")

    def load_data(
        self,
        level: str = "class",
        sample_file: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load the bug dataset

        Args:
            level: Either 'class' or 'file' level metrics
            sample_file: Optional path to a specific CSV file

        Returns:
            DataFrame with bug data
        """
        if sample_file:
            df = pd.read_csv(sample_file)
            print(f"Loaded custom dataset: {sample_file}")
        else:
            # Look for UnifiedBugDataset files
            search_paths = [
                # Direct paths
                self.data_path / f"Unified-{level}.csv",
                self.data_path / f"UnifiedBugDataset-1.2/UnifiedBugDataset-1.2/Unified-{level}.csv",
                self.data_path / f"UnifiedBugDataset-1.2/Unified-{level}.csv",
                # BugPrediction folder
                self.data_path / f"UnifiedBugDataset-1.2/UnifiedBugDataset-1.2/BugPrediction/BugPrediction-unified-{level}.csv",
                self.data_path / f"BugPrediction/BugPrediction-unified-{level}.csv",
                # Alternative patterns
                self.data_path / f"bug-metrics-{level}s.csv",
                self.data_path / f"bug-data-{level}.csv",
                self.data_path / "unified_bug_dataset.csv",
                self.data_path / "bug_dataset.csv",
                self.data_path / "sample_bug_dataset.csv"
            ]

            df = None
            for file_path in search_paths:
                if file_path.exists():
                    df = pd.read_csv(file_path)
                    print(f"Loaded dataset from: {file_path}")
                    print(f"\nDataset shape: {df.shape}")
                    print(f"Columns found: {len(df.columns)}")
                    print(f"\nAll columns:\n{list(df.columns)}")
                    break

            if df is None:
                # If no file found, create a sample dataset
                print("No dataset found in any expected location. Creating sample dataset...")
                print(f"Searched in: {self.data_path}")
                df = self._create_sample_dataset()

        # Store based on level
        if level == "class":
            self.df_classes = df
        else:
            self.df_files = df

        return df

    def _create_sample_dataset(self) -> pd.DataFrame:
        """
        Create a sample dataset for demonstration
        """
        np.random.seed(42)
        n_samples = 1000

        data = {
            'Name': [f'Class{i}' for i in range(n_samples)],
            'Path': [f'src/package{i//100}/Class{i}.java' for i in range(n_samples)],
            'LOC': np.random.randint(10, 500, n_samples),
            'LLOC': np.random.randint(5, 400, n_samples),
            'WMC': np.random.randint(1, 50, n_samples),
            'CBO': np.random.randint(0, 30, n_samples),
            'RFC': np.random.randint(1, 100, n_samples),
            'LCOM5': np.random.uniform(0, 1, n_samples),
            'DIT': np.random.randint(0, 10, n_samples),
            'NOC': np.random.randint(0, 20, n_samples),
            'McCabe': np.random.randint(1, 50, n_samples),
            'NOS': np.random.randint(0, 100, n_samples),
        }

        # Create bug labels (binary: 0 or 1)
        # Higher complexity = higher bug probability
        bug_prob = (
            (data['WMC'] > 20).astype(int) * 0.3 +
            (data['CBO'] > 15).astype(int) * 0.3 +
            (data['LOC'] > 200).astype(int) * 0.2 +
            (data['McCabe'] > 15).astype(int) * 0.2
        )
        data['bug'] = (np.random.random(n_samples) < bug_prob).astype(int)

        df = pd.DataFrame(data)

        # Save sample dataset
        sample_path = self.data_path / "sample_bug_dataset.csv"
        df.to_csv(sample_path, index=False)
        print(f"Sample dataset saved to: {sample_path}")

        return df

    def get_metrics_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of metric columns (excluding metadata and target)

        Args:
            df: Input DataFrame

        Returns:
            List of metric column names
        """
        exclude_cols = ['Name', 'Path', 'bug', 'Project', 'Version']
        return [col for col in df.columns if col not in exclude_cols]

    def prepare_features(
        self,
        df: pd.DataFrame,
        metrics: Optional[list] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target variable

        Args:
            df: Input DataFrame
            metrics: Optional list of specific metrics to use

        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        if metrics is None:
            metrics = self.get_metrics_columns(df)

        # Select only numeric columns
        X = df[metrics].select_dtypes(include=[np.number])

        # Handle missing values
        X = X.fillna(X.median())

        # Binary classification: 0 = no bugs, 1 = has bugs
        if 'bug' in df.columns:
            y = (df['bug'] > 0).astype(int)
        else:
            # If no bug column, create dummy target
            y = pd.Series(np.zeros(len(df)), dtype=int)

        return X, y

    def get_dataset_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the dataset

        Args:
            df: Input DataFrame

        Returns:
            Dictionary with dataset statistics
        """
        X, y = self.prepare_features(df)

        info = {
            'total_samples': len(df),
            'buggy_samples': y.sum(),
            'clean_samples': (y == 0).sum(),
            'bug_rate': y.mean(),
            'n_features': X.shape[1],
            'feature_names': list(X.columns),
            'missing_values': df.isnull().sum().sum(),
        }

        return info

    def print_dataset_info(self, df: pd.DataFrame) -> None:
        """
        Print dataset information

        Args:
            df: Input DataFrame
        """
        info = self.get_dataset_info(df)

        print("\n" + "="*70)
        print("DATASET INFORMATION")
        print("="*70)
        print(f"Total Samples: {info['total_samples']:,}")
        print(f"Buggy Samples: {info['buggy_samples']:,} ({info['bug_rate']:.2%})")
        print(f"Clean Samples: {info['clean_samples']:,} ({1-info['bug_rate']:.2%})")
        print(f"Number of Features: {info['n_features']}")
        print(f"Missing Values: {info['missing_values']}")

        print("\n" + "="*70)
        print("COLUMN INFORMATION")
        print("="*70)

        # Metadata columns
        metadata_cols = ['Name', 'Path', 'Project', 'Version']
        existing_metadata = [col for col in metadata_cols if col in df.columns]
        if existing_metadata:
            print(f"\nMetadata Columns ({len(existing_metadata)}):")
            for col in existing_metadata:
                print(f"  - {col}")

        # Target column
        if 'bug' in df.columns:
            print(f"\nTarget Column:")
            print(f"  - bug (min: {df['bug'].min()}, max: {df['bug'].max()}, mean: {df['bug'].mean():.2f})")

        # Feature columns
        feature_cols = info['feature_names']
        print(f"\nFeature Columns ({len(feature_cols)}):")

        # Group features by type
        size_metrics = [c for c in feature_cols if any(x in c.upper() for x in ['LOC', 'LLOC', 'TLOC', 'TLLOC', 'CLOC'])]
        complexity_metrics = [c for c in feature_cols if any(x in c.upper() for x in ['WMC', 'MCCABE', 'CC', 'NL', 'NLE'])]
        coupling_metrics = [c for c in feature_cols if any(x in c.upper() for x in ['CBO', 'RFC', 'NII', 'NOI', 'CBOI'])]
        cohesion_metrics = [c for c in feature_cols if any(x in c.upper() for x in ['LCOM', 'TCC', 'LCC'])]
        inheritance_metrics = [c for c in feature_cols if any(x in c.upper() for x in ['DIT', 'NOC', 'NOA', 'NOM', 'NOP', 'NOD'])]

        if size_metrics:
            print(f"\n  Size Metrics ({len(size_metrics)}):")
            for col in size_metrics[:10]:  # Show first 10
                print(f"    - {col}")
            if len(size_metrics) > 10:
                print(f"    ... and {len(size_metrics) - 10} more")

        if complexity_metrics:
            print(f"\n  Complexity Metrics ({len(complexity_metrics)}):")
            for col in complexity_metrics[:10]:
                print(f"    - {col}")
            if len(complexity_metrics) > 10:
                print(f"    ... and {len(complexity_metrics) - 10} more")

        if coupling_metrics:
            print(f"\n  Coupling Metrics ({len(coupling_metrics)}):")
            for col in coupling_metrics[:10]:
                print(f"    - {col}")
            if len(coupling_metrics) > 10:
                print(f"    ... and {len(coupling_metrics) - 10} more")

        if cohesion_metrics:
            print(f"\n  Cohesion Metrics ({len(cohesion_metrics)}):")
            for col in cohesion_metrics[:10]:
                print(f"    - {col}")
            if len(cohesion_metrics) > 10:
                print(f"    ... and {len(cohesion_metrics) - 10} more")

        if inheritance_metrics:
            print(f"\n  Inheritance Metrics ({len(inheritance_metrics)}):")
            for col in inheritance_metrics[:10]:
                print(f"    - {col}")
            if len(inheritance_metrics) > 10:
                print(f"    ... and {len(inheritance_metrics) - 10} more")

        # Other metrics
        categorized = set(size_metrics + complexity_metrics + coupling_metrics + cohesion_metrics + inheritance_metrics)
        other_metrics = [c for c in feature_cols if c not in categorized]

        if other_metrics:
            print(f"\n  Other Metrics ({len(other_metrics)}):")
            for col in other_metrics[:10]:
                print(f"    - {col}")
            if len(other_metrics) > 10:
                print(f"    ... and {len(other_metrics) - 10} more")

        # Summary statistics for key metrics
        print("\n" + "="*70)
        print("KEY METRICS SUMMARY")
        print("="*70)

        key_metrics = ['LOC', 'WMC', 'CBO', 'RFC', 'LCOM5', 'DIT', 'NOC', 'McCabe']
        available_key_metrics = [m for m in key_metrics if m in df.columns]

        if available_key_metrics:
            print(f"\n{'Metric':<15} {'Min':<10} {'Max':<10} {'Mean':<10} {'Median':<10}")
            print("-" * 70)
            for metric in available_key_metrics:
                values = df[metric]
                print(f"{metric:<15} {values.min():<10.2f} {values.max():<10.2f} {values.mean():<10.2f} {values.median():<10.2f}")

        print("\n" + "="*70)
        print(f"\nAll {len(feature_cols)} Features:")
        print(", ".join(feature_cols))
        print("="*70 + "\n")


def download_dataset(data_path: str = "../data"):
    """
    Convenience function to download the dataset

    Args:
        data_path: Path to save the dataset
    """
    dataset = BugDataset(data_path)
    dataset.download_dataset()


if __name__ == "__main__":
    # Example usage
    dataset = BugDataset()

    # Try to load existing dataset or create sample
    df = dataset.load_data(level="class")

    # Print information
    dataset.print_dataset_info(df)

    # Prepare features
    X, y = dataset.prepare_features(df)
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")