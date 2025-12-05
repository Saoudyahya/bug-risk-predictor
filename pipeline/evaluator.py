"""
Model evaluation pipeline
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import joblib

from utils.metrics import evaluate_model, MetricsCalculator


class ModelEvaluator:
    """
    Comprehensive model evaluation
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the saved model
        """
        self.model_path = Path(model_path)
        self.model = None
        self.model_type = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the saved model"""
        print(f"Loading model from: {self.model_path}")
        
        try:
            # Try loading as joblib
            model_data = joblib.load(self.model_path)
            
            if isinstance(model_data, dict):
                if 'model' in model_data:
                    # Custom model format
                    self.model = model_data
                    self.model_type = 'custom'
                else:
                    # sklearn model format
                    self.model = model_data
                    self.model_type = 'sklearn'
            else:
                # Direct model object
                self.model = model_data
                self.model_type = 'direct'
            
            print(f"Model loaded successfully (type: {self.model_type})")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> tuple:
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model_type == 'custom':
            if 'model' in self.model:
                # Custom wrapped model
                model_obj = self.model['model']
                if hasattr(model_obj, 'predict'):
                    y_pred = model_obj.predict(X)
                    y_proba = model_obj.predict_proba(X) if hasattr(model_obj, 'predict_proba') else None
                else:
                    raise ValueError("Model does not have predict method")
            else:
                raise ValueError("Invalid custom model format")
        
        elif self.model_type == 'sklearn':
            model_obj = self.model['model']
            y_pred = model_obj.predict(X)
            y_proba = model_obj.predict_proba(X) if hasattr(model_obj, 'predict_proba') else None
        
        else:
            # Direct model
            if hasattr(self.model, 'predict'):
                y_pred = self.model.predict(X)
                y_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None
            else:
                raise ValueError("Model does not have predict method")
        
        return y_pred, y_proba
    
    def evaluate_on_data(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: list = None,
        plot: bool = True
    ) -> dict:
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            plot: Whether to create plots
            
        Returns:
            Dictionary of evaluation results
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Make predictions
        y_pred, y_proba = self.predict(X_test)
        
        # Get feature importance if available
        feature_importance = None
        if self.model_type in ['custom', 'sklearn']:
            feature_importance = self.model.get('feature_importance')
        
        # Evaluate
        results = evaluate_model(
            y_test,
            y_pred,
            y_proba,
            feature_names=feature_names,
            feature_importance=feature_importance,
            model_name=self.model_path.stem,
            plot=plot
        )
        
        return results
    
    def evaluate_on_file(
        self,
        data_path: str,
        target_column: str = 'bug',
        plot: bool = True
    ) -> dict:
        """
        Evaluate model on data from file
        
        Args:
            data_path: Path to CSV file with test data
            target_column: Name of target column
            plot: Whether to create plots
            
        Returns:
            Dictionary of evaluation results
        """
        print(f"\nLoading test data from: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Separate features and target
        y_test = (df[target_column] > 0).astype(int)
        
        # Get feature columns
        exclude_cols = ['Name', 'Path', target_column, 'Project', 'Version']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X_test = df[feature_cols].fillna(0)
        
        print(f"Test data shape: {X_test.shape}")
        
        # Evaluate
        return self.evaluate_on_data(X_test.values, y_test.values, feature_cols, plot)
    
    def predict_from_metrics(
        self,
        metrics_dict: dict,
        feature_names: list = None
    ) -> dict:
        """
        Predict from a dictionary of metrics
        
        Args:
            metrics_dict: Dictionary of metric values
            feature_names: List of expected feature names (for ordering)
            
        Returns:
            Dictionary with prediction results
        """
        # Convert metrics to array
        if feature_names:
            X = np.array([[metrics_dict.get(fname, 0) for fname in feature_names]])
        else:
            X = np.array([list(metrics_dict.values())])
        
        # Predict
        y_pred, y_proba = self.predict(X)
        
        # Prepare result
        result = {
            'prediction': int(y_pred[0]),
            'bug_probability': float(y_proba[0][1]) if y_proba is not None else None,
            'clean_probability': float(y_proba[0][0]) if y_proba is not None else None,
            'risk_level': self._get_risk_level(y_proba[0][1] if y_proba is not None else 0.5)
        }
        
        return result
    
    def _get_risk_level(self, probability: float) -> str:
        """Get risk level from probability"""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.8:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def batch_predict_from_file(
        self,
        input_file: str,
        output_file: str = None
    ) -> pd.DataFrame:
        """
        Make batch predictions from a file
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save results (optional)
            
        Returns:
            DataFrame with predictions
        """
        print(f"\nLoading data from: {input_file}")
        
        df = pd.read_csv(input_file)
        
        # Get feature columns
        exclude_cols = ['Name', 'Path', 'bug', 'Project', 'Version']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].fillna(0).values
        
        # Predict
        y_pred, y_proba = self.predict(X)
        
        # Add predictions to dataframe
        df['predicted_bug'] = y_pred
        if y_proba is not None:
            df['bug_probability'] = y_proba[:, 1]
            df['risk_level'] = df['bug_probability'].apply(self._get_risk_level)
        
        # Sort by bug probability
        if 'bug_probability' in df.columns:
            df = df.sort_values('bug_probability', ascending=False)
        
        # Save if output file specified
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Results saved to: {output_file}")
        
        return df
    
    def get_high_risk_files(
        self,
        input_file: str,
        top_n: int = 20,
        threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        Get high-risk files from predictions
        
        Args:
            input_file: Path to input CSV file
            top_n: Number of top risky files to return
            threshold: Minimum probability threshold
            
        Returns:
            DataFrame with high-risk files
        """
        df_predictions = self.batch_predict_from_file(input_file)
        
        # Filter high-risk files
        if 'bug_probability' in df_predictions.columns:
            high_risk = df_predictions[df_predictions['bug_probability'] >= threshold]
            high_risk = high_risk.head(top_n)
        else:
            high_risk = df_predictions[df_predictions['predicted_bug'] == 1].head(top_n)
        
        print(f"\nFound {len(high_risk)} high-risk files")
        
        return high_risk


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate bug prediction model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to saved model')
    parser.add_argument('--test-data', type=str,
                       help='Path to test data CSV file')
    parser.add_argument('--batch-predict', type=str,
                       help='Path to CSV file for batch prediction')
    parser.add_argument('--output', type=str,
                       help='Path to save prediction results')
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not create plots')
    parser.add_argument('--top-n', type=int, default=20,
                       help='Number of top risky files to show')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(args.model)
    
    if args.test_data:
        # Evaluate on test data
        results = evaluator.evaluate_on_file(
            args.test_data,
            plot=not args.no_plot
        )
        
    elif args.batch_predict:
        # Batch prediction
        df_results = evaluator.batch_predict_from_file(
            args.batch_predict,
            args.output
        )
        
        print(f"\nPrediction Results:")
        print(df_results[['Name', 'predicted_bug', 'bug_probability', 'risk_level']].head(20))
        
        # Show high-risk files
        high_risk = evaluator.get_high_risk_files(args.batch_predict, top_n=args.top_n)
        
        print(f"\nTop {args.top_n} High-Risk Files:")
        print(high_risk[['Name', 'Path', 'bug_probability', 'risk_level']])
        
    else:
        print("Please specify either --test-data or --batch-predict")
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
