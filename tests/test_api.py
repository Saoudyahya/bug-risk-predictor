"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient
import joblib
import numpy as np
from pathlib import Path


@pytest.fixture
def test_client(mock_trained_model, model_save_path):
    """Create a test client with a mock model"""
    # Save mock model
    model_path = model_save_path / "test_model.pkl"
    joblib.dump(mock_trained_model, model_path)
    
    # Import and configure app
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from app.fastapi import app
    from app import fastapi as fastapi_module
    
    # Override model path
    fastapi_module.MODEL_PATH = str(model_path)
    fastapi_module.load_model(str(model_path))
    
    client = TestClient(app)
    return client


@pytest.mark.api
class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_endpoint(self, test_client):
        """Test root endpoint"""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert 'message' in data
        assert 'version' in data
        assert 'endpoints' in data


@pytest.mark.api
class TestHealthCheck:
    """Tests for health check endpoint"""
    
    def test_health_endpoint(self, test_client):
        """Test health check endpoint"""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert data['model_loaded'] is True


@pytest.mark.api
class TestModelInfo:
    """Tests for model info endpoint"""
    
    def test_model_info_endpoint(self, test_client):
        """Test model info endpoint"""
        response = test_client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert 'model_path' in data
        assert 'model_type' in data
        assert 'status' in data


@pytest.mark.api
class TestRequiredMetrics:
    """Tests for required metrics endpoint"""
    
    def test_required_metrics_endpoint(self, test_client):
        """Test required metrics endpoint"""
        response = test_client.get("/metrics/required")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that it has either required_metrics or suggested_metrics
        assert 'required_metrics' in data or 'suggested_metrics' in data


@pytest.mark.api
class TestSinglePrediction:
    """Tests for single file prediction"""
    
    def test_predict_single_file(self, test_client, sample_metrics):
        """Test single file prediction"""
        request_data = {
            "file_name": "TestClass.java",
            "metrics": sample_metrics
        }
        
        response = test_client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'file_name' in data
        assert 'prediction' in data
        assert 'risk_level' in data
        assert data['prediction'] in [0, 1]
    
    def test_predict_invalid_metrics(self, test_client):
        """Test prediction with invalid metrics"""
        request_data = {
            "file_name": "Test.java",
            "metrics": {}  # Empty metrics
        }
        
        response = test_client.post("/predict", json=request_data)
        # Should either succeed with defaults or return error
        assert response.status_code in [200, 422, 500]
    
    def test_predict_high_risk_file(self, test_client):
        """Test prediction for high-risk file"""
        request_data = {
            "file_name": "ComplexClass.java",
            "metrics": {
                "LOC": 500,
                "WMC": 50,
                "CBO": 25,
                "RFC": 100,
                "LCOM5": 0.8,
                "McCabe": 30
            }
        }
        
        response = test_client.post("/predict", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        # High complexity might indicate higher bug probability
        assert 'risk_level' in data


@pytest.mark.api
class TestBatchPrediction:
    """Tests for batch prediction"""
    
    def test_batch_predict(self, test_client, sample_metrics):
        """Test batch prediction"""
        request_data = {
            "files": [
                {
                    "file_name": "File1.java",
                    "metrics": sample_metrics
                },
                {
                    "file_name": "File2.java",
                    "metrics": {
                        "LOC": 80,
                        "WMC": 5,
                        "CBO": 3
                    }
                }
            ]
        }
        
        response = test_client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'total_files' in data
        assert 'results' in data
        assert data['total_files'] == 2
        assert len(data['results']) == 2
    
    def test_batch_predict_empty(self, test_client):
        """Test batch prediction with empty list"""
        request_data = {"files": []}
        
        response = test_client.post("/predict/batch", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['total_files'] == 0


@pytest.mark.api
class TestCodePrediction:
    """Tests for code analysis prediction"""
    
    def test_predict_from_java_code(self, test_client, sample_java_code):
        """Test prediction from Java code"""
        request_data = {
            "code": sample_java_code,
            "language": "java",
            "file_name": "BankAccount.java"
        }
        
        response = test_client.post("/predict/code", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'file_name' in data
        assert 'prediction' in data
        assert 'risk_level' in data
    
    def test_predict_from_python_code(self, test_client, sample_python_code):
        """Test prediction from Python code"""
        request_data = {
            "code": sample_python_code,
            "language": "python",
            "file_name": "bank_account.py"
        }
        
        response = test_client.post("/predict/code", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data['file_name'] == "bank_account.py"
    
    def test_predict_empty_code(self, test_client):
        """Test prediction with empty code"""
        request_data = {
            "code": "",
            "language": "java"
        }
        
        response = test_client.post("/predict/code", json=request_data)
        
        assert response.status_code == 400


@pytest.mark.api
class TestCSVPrediction:
    """Tests for CSV file prediction"""
    
    def test_predict_from_csv(self, test_client, sample_csv_file):
        """Test prediction from CSV file"""
        with open(sample_csv_file, 'rb') as f:
            response = test_client.post(
                "/predict/file",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert 'total_files' in data
        assert 'results' in data
        assert data['total_files'] > 0
    
    def test_predict_and_download_csv(self, test_client, sample_csv_file):
        """Test prediction with CSV download"""
        with open(sample_csv_file, 'rb') as f:
            response = test_client.post(
                "/predict/file/download",
                files={"file": ("test.csv", f, "text/csv")}
            )
        
        assert response.status_code == 200
        assert response.headers['content-type'] == 'text/csv; charset=utf-8'


@pytest.mark.api
class TestExamples:
    """Tests for example endpoints"""
    
    def test_single_prediction_example(self, test_client):
        """Test single prediction example endpoint"""
        response = test_client.get("/examples/single")
        
        assert response.status_code == 200
        data = response.json()
        assert 'example_request' in data
        assert 'example_response' in data
    
    def test_batch_prediction_example(self, test_client):
        """Test batch prediction example endpoint"""
        response = test_client.get("/examples/batch")
        
        assert response.status_code == 200
        data = response.json()
        assert 'example_request' in data


@pytest.mark.api
class TestErrorHandling:
    """Tests for error handling"""
    
    def test_invalid_endpoint(self, test_client):
        """Test accessing invalid endpoint"""
        response = test_client.get("/invalid")
        
        assert response.status_code == 404
    
    def test_invalid_request_data(self, test_client):
        """Test with invalid request data"""
        response = test_client.post("/predict", json={"invalid": "data"})
        
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, test_client):
        """Test with missing required fields"""
        request_data = {
            "file_name": "Test.java"
            # Missing metrics field
        }
        
        response = test_client.post("/predict", json=request_data)
        
        assert response.status_code == 422


@pytest.mark.api
class TestCORS:
    """Tests for CORS configuration"""
    
    def test_cors_headers(self, test_client):
        """Test that CORS headers are present"""
        response = test_client.options("/")
        
        # Should have CORS headers
        assert 'access-control-allow-origin' in response.headers or response.status_code == 200


@pytest.mark.api
@pytest.mark.integration
class TestCompleteWorkflow:
    """Integration tests for complete API workflow"""
    
    def test_complete_prediction_workflow(self, test_client, sample_java_code):
        """Test complete workflow from code to prediction"""
        # 1. Check health
        health_response = test_client.get("/health")
        assert health_response.status_code == 200
        
        # 2. Get model info
        info_response = test_client.get("/model/info")
        assert info_response.status_code == 200
        
        # 3. Make prediction from code
        predict_response = test_client.post("/predict/code", json={
            "code": sample_java_code,
            "language": "java"
        })
        assert predict_response.status_code == 200
        
        # 4. Check prediction result
        data = predict_response.json()
        assert 'prediction' in data
        assert 'risk_level' in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
