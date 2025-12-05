#!/usr/bin/env python3
"""
Test Runner Script for Bug Prediction System

This script provides convenient commands for running different types of tests.
"""

import subprocess
import sys
import argparse
from pathlib import Path


class TestRunner:
    """Test runner with various test configurations"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.tests_dir = self.project_root / "tests"
    
    def run_command(self, command):
        """Run a shell command"""
        print(f"\n{'='*70}")
        print(f"Running: {' '.join(command)}")
        print(f"{'='*70}\n")
        
        result = subprocess.run(command, cwd=self.project_root)
        return result.returncode
    
    def run_all_tests(self, verbose=True):
        """Run all tests"""
        command = ["pytest"]
        if verbose:
            command.append("-v")
        return self.run_command(command)
    
    def run_unit_tests(self):
        """Run only unit tests"""
        command = ["pytest", "-m", "unit", "-v"]
        return self.run_command(command)
    
    def run_integration_tests(self):
        """Run only integration tests"""
        command = ["pytest", "-m", "integration", "-v"]
        return self.run_command(command)
    
    def run_api_tests(self):
        """Run API tests"""
        command = ["pytest", "-m", "api", "-v"]
        return self.run_command(command)
    
    def run_model_tests(self):
        """Run model tests"""
        command = ["pytest", "-m", "model", "-v"]
        return self.run_command(command)
    
    def run_preprocessing_tests(self):
        """Run preprocessing tests"""
        command = ["pytest", "-m", "preprocessing", "-v"]
        return self.run_command(command)
    
    def run_fast_tests(self):
        """Run fast tests (exclude slow tests)"""
        command = ["pytest", "-m", "not slow", "-v"]
        return self.run_command(command)
    
    def run_with_coverage(self):
        """Run tests with coverage report"""
        command = [
            "pytest",
            "--cov=app",
            "--cov=core",
            "--cov=models",
            "--cov=pipeline",
            "--cov=utils",
            "--cov-report=html",
            "--cov-report=term-missing",
            "-v"
        ]
        return self.run_command(command)
    
    def run_parallel(self, num_workers=4):
        """Run tests in parallel"""
        command = ["pytest", "-n", str(num_workers), "-v"]
        return self.run_command(command)
    
    def run_specific_file(self, test_file):
        """Run tests in a specific file"""
        command = ["pytest", str(self.tests_dir / test_file), "-v"]
        return self.run_command(command)
    
    def run_specific_test(self, test_name):
        """Run a specific test by name"""
        command = ["pytest", "-k", test_name, "-v"]
        return self.run_command(command)
    
    def run_with_html_report(self):
        """Generate HTML test report"""
        command = [
            "pytest",
            "--html=test-report.html",
            "--self-contained-html",
            "-v"
        ]
        return self.run_command(command)
    
    def check_coverage(self):
        """Check test coverage and fail if below threshold"""
        command = [
            "pytest",
            "--cov=app",
            "--cov=core",
            "--cov=models",
            "--cov=pipeline",
            "--cov=utils",
            "--cov-report=term-missing",
            "--cov-fail-under=70",  # Fail if coverage < 70%
            "-v"
        ]
        return self.run_command(command)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Test Runner for Bug Prediction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all              # Run all tests
  python run_tests.py --unit             # Run unit tests only
  python run_tests.py --api              # Run API tests only
  python run_tests.py --coverage         # Run with coverage report
  python run_tests.py --parallel         # Run tests in parallel
  python run_tests.py --file test_api.py # Run specific test file
  python run_tests.py --test test_predict # Run tests matching pattern
  python run_tests.py --fast             # Run fast tests only
  python run_tests.py --html             # Generate HTML report
        """
    )
    
    parser.add_argument('--all', action='store_true',
                       help='Run all tests')
    parser.add_argument('--unit', action='store_true',
                       help='Run unit tests only')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests only')
    parser.add_argument('--api', action='store_true',
                       help='Run API tests only')
    parser.add_argument('--model', action='store_true',
                       help='Run model tests only')
    parser.add_argument('--preprocessing', action='store_true',
                       help='Run preprocessing tests only')
    parser.add_argument('--fast', action='store_true',
                       help='Run fast tests (exclude slow tests)')
    parser.add_argument('--coverage', action='store_true',
                       help='Run tests with coverage report')
    parser.add_argument('--parallel', action='store_true',
                       help='Run tests in parallel')
    parser.add_argument('--html', action='store_true',
                       help='Generate HTML test report')
    parser.add_argument('--check-coverage', action='store_true',
                       help='Check coverage and fail if below threshold')
    parser.add_argument('--file', type=str,
                       help='Run specific test file')
    parser.add_argument('--test', type=str,
                       help='Run specific test by name pattern')
    parser.add_argument('--workers', type=int, default=4,
                       help='Number of parallel workers (default: 4)')
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Default to running all tests if no specific option is provided
    if not any([
        args.all, args.unit, args.integration, args.api, args.model,
        args.preprocessing, args.fast, args.coverage, args.parallel,
        args.html, args.check_coverage, args.file, args.test
    ]):
        args.all = True
    
    exit_code = 0
    
    try:
        if args.check_coverage:
            exit_code = runner.check_coverage()
        elif args.coverage:
            exit_code = runner.run_with_coverage()
        elif args.parallel:
            exit_code = runner.run_parallel(args.workers)
        elif args.html:
            exit_code = runner.run_with_html_report()
        elif args.unit:
            exit_code = runner.run_unit_tests()
        elif args.integration:
            exit_code = runner.run_integration_tests()
        elif args.api:
            exit_code = runner.run_api_tests()
        elif args.model:
            exit_code = runner.run_model_tests()
        elif args.preprocessing:
            exit_code = runner.run_preprocessing_tests()
        elif args.fast:
            exit_code = runner.run_fast_tests()
        elif args.file:
            exit_code = runner.run_specific_file(args.file)
        elif args.test:
            exit_code = runner.run_specific_test(args.test)
        elif args.all:
            exit_code = runner.run_all_tests()
        
        if exit_code == 0:
            print("\n" + "="*70)
            print("✅ ALL TESTS PASSED!")
            print("="*70 + "\n")
        else:
            print("\n" + "="*70)
            print("❌ SOME TESTS FAILED!")
            print("="*70 + "\n")
    
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"\n\nError running tests: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
