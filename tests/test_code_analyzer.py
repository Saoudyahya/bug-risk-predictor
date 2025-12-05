"""
Tests for code analyzer module
"""

import pytest
from app.code_analyzer import (
    analyze_java_code,
    analyze_python_code,
    analyze_code,
    CodeAnalysisRequest
)
from fastapi import HTTPException


@pytest.mark.unit
class TestJavaCodeAnalyzer:
    """Tests for Java code analysis"""
    
    def test_analyze_simple_java_class(self, sample_java_code):
        """Test analysis of simple Java class"""
        metrics = analyze_java_code(sample_java_code)
        
        assert 'LOC' in metrics
        assert 'WMC' in metrics
        assert 'McCabe' in metrics
        assert 'CBO' in metrics
        
        # Check reasonable values
        assert metrics['LOC'] > 0
        assert metrics['WMC'] >= 3  # At least 3 methods
        assert metrics['McCabe'] > 0
    
    def test_analyze_empty_java_code(self):
        """Test analysis of empty code"""
        code = ""
        metrics = analyze_java_code(code)
        
        assert metrics['LOC'] == 0
        assert metrics['WMC'] == 0
    
    def test_analyze_java_with_inheritance(self):
        """Test analysis of Java class with inheritance"""
        code = """
        public class ChildClass extends ParentClass {
            private int value;
            
            public void method() {
                // implementation
            }
        }
        """
        metrics = analyze_java_code(code)
        
        assert metrics['DIT'] == 1  # Has extends keyword
    
    def test_analyze_java_complex_class(self):
        """Test analysis of complex Java class"""
        code = """
        import java.util.List;
        import java.util.ArrayList;
        
        public class ComplexClass {
            private int value;
            
            public void method1() {
                if (value > 0) {
                    for (int i = 0; i < 10; i++) {
                        // loop
                    }
                }
            }
            
            public void method2() {
                while (value < 100) {
                    value++;
                }
            }
            
            public void method3() {
                try {
                    // some code
                } catch (Exception e) {
                    // handle
                }
            }
        }
        """
        metrics = analyze_java_code(code)
        
        assert metrics['WMC'] >= 3
        assert metrics['McCabe'] > 3  # Multiple decision points
        assert metrics['CBO'] > 0  # Has imports


@pytest.mark.unit
class TestPythonCodeAnalyzer:
    """Tests for Python code analysis"""
    
    def test_analyze_simple_python_class(self, sample_python_code):
        """Test analysis of simple Python class"""
        metrics = analyze_python_code(sample_python_code)
        
        assert 'LOC' in metrics
        assert 'WMC' in metrics
        assert 'McCabe' in metrics
        
        assert metrics['LOC'] > 0
        assert metrics['WMC'] >= 4  # __init__ + 3 methods
    
    def test_analyze_python_with_imports(self):
        """Test analysis of Python code with imports"""
        code = """
        import numpy as np
        from sklearn import metrics
        
        class DataProcessor:
            def process(self):
                pass
        """
        metrics = analyze_python_code(code)
        
        assert metrics['CBO'] > 0  # Has imports
    
    def test_analyze_python_complex_logic(self):
        """Test analysis of Python with complex logic"""
        code = """
        def complex_function():
            if condition1:
                for i in range(10):
                    if condition2:
                        while condition3:
                            pass
                    elif condition4:
                        pass
        """
        metrics = analyze_python_code(code)
        
        assert metrics['McCabe'] > 5  # Multiple decision points


@pytest.mark.unit
class TestCodeAnalyzer:
    """Tests for generic code analyzer"""
    
    def test_analyze_java_code(self, sample_java_code):
        """Test Java code analysis through generic function"""
        metrics = analyze_code(sample_java_code, language="java")
        
        assert isinstance(metrics, dict)
        assert 'LOC' in metrics
        assert metrics['LOC'] > 0
    
    def test_analyze_python_code(self, sample_python_code):
        """Test Python code analysis through generic function"""
        metrics = analyze_code(sample_python_code, language="python")
        
        assert isinstance(metrics, dict)
        assert 'LOC' in metrics
        assert metrics['LOC'] > 0
    
    def test_analyze_unknown_language(self):
        """Test analysis of unknown language"""
        code = "some code"
        metrics = analyze_code(code, language="javascript")
        
        # Should return default metrics
        assert isinstance(metrics, dict)
        assert 'LOC' in metrics
    
    def test_analyze_empty_code(self):
        """Test analysis of empty code"""
        with pytest.raises(HTTPException) as exc_info:
            analyze_code("", language="java")
        
        assert exc_info.value.status_code == 400
    
    def test_analyze_very_short_code(self):
        """Test analysis of very short code"""
        with pytest.raises(HTTPException):
            analyze_code("x = 1", language="python")


@pytest.mark.unit
class TestCodeAnalysisRequest:
    """Tests for CodeAnalysisRequest model"""
    
    def test_create_request(self):
        """Test creating a code analysis request"""
        request = CodeAnalysisRequest(
            code="public class Test {}",
            language="java",
            file_name="Test.java"
        )
        
        assert request.code == "public class Test {}"
        assert request.language == "java"
        assert request.file_name == "Test.java"
    
    def test_default_values(self):
        """Test default values in request"""
        request = CodeAnalysisRequest(code="some code")
        
        assert request.language == "java"
        assert request.file_name == "Code.java"


@pytest.mark.unit
class TestMetricsValidation:
    """Tests for metrics validation"""
    
    def test_all_required_metrics_present(self, sample_java_code):
        """Test that all required metrics are present"""
        metrics = analyze_java_code(sample_java_code)
        
        required_metrics = [
            'LOC', 'LLOC', 'NOS', 'WMC', 'McCabe',
            'CBO', 'RFC', 'LCOM5', 'DIT', 'NOC'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing metric: {metric}"
    
    def test_metrics_have_valid_types(self, sample_java_code):
        """Test that metrics have valid types"""
        metrics = analyze_java_code(sample_java_code)
        
        assert isinstance(metrics['LOC'], int)
        assert isinstance(metrics['LLOC'], int)
        assert isinstance(metrics['WMC'], int)
        assert isinstance(metrics['LCOM5'], float)
        assert isinstance(metrics['CBO'], int)
    
    def test_metrics_have_valid_ranges(self, sample_java_code):
        """Test that metrics are within valid ranges"""
        metrics = analyze_java_code(sample_java_code)
        
        assert metrics['LOC'] >= 0
        assert metrics['LLOC'] >= 0
        assert metrics['WMC'] >= 0
        assert 0 <= metrics['LCOM5'] <= 1
        assert metrics['McCabe'] >= 1  # Minimum complexity is 1
        assert metrics['CBO'] >= 0
        assert metrics['CBO'] <= 30  # Should be capped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
