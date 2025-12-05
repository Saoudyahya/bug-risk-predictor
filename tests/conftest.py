"""
Pytest configuration and shared fixtures
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# tests/conftest.py
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture(scope="session")
def sample_data():
    """Generate sample classification data"""
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.7, 0.3],
        random_state=42
    )
    return X, y


@pytest.fixture(scope="session")
def sample_dataframe(sample_data):
    """Generate sample DataFrame with bug data"""
    X, y = sample_data
    
    feature_names = [
        'LOC', 'LLOC', 'WMC', 'CBO', 'RFC', 'LCOM5', 
        'DIT', 'NOC', 'McCabe', 'NOS', 'NL', 'NLE',
        'TLOC', 'TLLOC', 'NII', 'NOI', 'CBOI', 'TCC', 'LCC', 'NOA'
    ]
    
    df = pd.DataFrame(X, columns=feature_names)
    df['bug'] = y
    df['Name'] = [f'Class{i}' for i in range(len(df))]
    df['Path'] = [f'src/package/Class{i}.java' for i in range(len(df))]
    
    return df


@pytest.fixture(scope="session")
def train_test_data(sample_data):
    """Split data into train and test sets"""
    X, y = sample_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def sample_csv_file(temp_dir, sample_dataframe):
    """Create a sample CSV file"""
    csv_path = temp_dir / "sample_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_metrics():
    """Sample software metrics for testing"""
    return {
        'LOC': 150,
        'LLOC': 120,
        'WMC': 12,
        'CBO': 5,
        'RFC': 25,
        'LCOM5': 0.3,
        'DIT': 2,
        'NOC': 0,
        'McCabe': 8,
        'NOS': 50
    }


@pytest.fixture
def sample_java_code():
    """Sample Java code for testing"""
    return """
public class BankAccount {
    private double balance;
    private String accountNumber;
    
    public BankAccount(String accountNumber) {
        this.accountNumber = accountNumber;
        this.balance = 0.0;
    }
    
    public void deposit(double amount) {
        if (amount > 0) {
            balance += amount;
        }
    }
    
    public boolean withdraw(double amount) {
        if (amount > 0 && amount <= balance) {
            balance -= amount;
            return true;
        }
        return false;
    }
    
    public double getBalance() {
        return balance;
    }
}
"""


@pytest.fixture
def sample_python_code():
    """Sample Python code for testing"""
    return """
class BankAccount:
    def __init__(self, account_number):
        self.account_number = account_number
        self.balance = 0.0
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
    
    def withdraw(self, amount):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self):
        return self.balance
"""


@pytest.fixture
def mock_trained_model():
    """Create a mock trained model for testing"""
    from sklearn.ensemble import RandomForestClassifier
    
    # Create and train a simple model
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42
    )
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    
    feature_names = [f'Feature_{i}' for i in range(10)]
    
    return {
        'model': model,
        'feature_names': feature_names,
        'feature_importance': model.feature_importances_
    }


@pytest.fixture
def model_save_path(temp_dir):
    """Path for saving test models"""
    model_path = temp_dir / "models"
    model_path.mkdir(exist_ok=True)
    return model_path


# Pytest configuration
def pytest_configure(config):
    """Configure pytest"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "api: API tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "model: Model training tests")
    config.addinivalue_line("markers", "preprocessing: Preprocessing tests")
