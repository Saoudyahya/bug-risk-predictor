"""
FastAPI REST API for bug prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import io

from pipeline.evaluator import ModelEvaluator
from app.code_analyzer import analyze_code, CodeAnalysisRequest

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler"""
    # Startup
    print("\n🚀 Starting Bug Prediction API...")
    if load_model(MODEL_PATH):
        print("✅ API ready!")
    else:
        print("⚠️  API started but model not loaded. Please load manually.")

    yield

    # Shutdown
    print("\n👋 Shutting down API...")

# Initialize FastAPI app
app = FastAPI(
    title="Bug Prediction API",
    description="AI-powered software bug prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model evaluator
evaluator = None
MODEL_PATH = "models/random_forest_model.pkl"


# Pydantic models for request/response validation
class MetricsInput(BaseModel):
    """Software metrics input"""
    LOC: Optional[float] = Field(None, description="Lines of Code")
    LLOC: Optional[float] = Field(None, description="Logical Lines of Code")
    WMC: Optional[float] = Field(None, description="Weighted Methods per Class")
    CBO: Optional[float] = Field(None, description="Coupling Between Objects")
    RFC: Optional[float] = Field(None, description="Response for Class")
    LCOM5: Optional[float] = Field(None, description="Lack of Cohesion in Methods")
    DIT: Optional[float] = Field(None, description="Depth of Inheritance Tree")
    NOC: Optional[float] = Field(None, description="Number of Children")
    McCabe: Optional[float] = Field(None, description="Cyclomatic Complexity")
    NOS: Optional[float] = Field(None, description="Number of Statements")

    model_config = {"extra": "allow"}  # Pydantic V2 style


class PredictionRequest(BaseModel):
    """Single file prediction request"""
    file_name: str = Field(..., description="Name of the file/class")
    metrics: Dict[str, float] = Field(..., description="Software metrics dictionary")


class FileMetrics(BaseModel):
    """File with metrics"""
    file_name: str
    metrics: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    files: List[FileMetrics] = Field(..., description="List of files with metrics")


class PredictionResponse(BaseModel):
    """Prediction response"""
    file_name: str
    prediction: int = Field(..., description="0 = Clean, 1 = Buggy")
    bug_probability: Optional[float] = Field(None, description="Probability of bug (0-1)")
    clean_probability: Optional[float] = Field(None, description="Probability of no bug (0-1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    confidence: Optional[float] = Field(None, description="Prediction confidence (0-1)")
    top_risk_factors: Optional[List[Dict[str, float]]] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_files: int
    high_risk_files: int
    critical_risk_files: int
    results: List[PredictionResponse]


class ModelInfo(BaseModel):
    """Model information"""
    model_path: str
    model_type: str
    status: str
    feature_count: Optional[int] = None
    required_metrics: Optional[List[str]] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    api_version: str


def load_model(model_path: str = MODEL_PATH):
    """Load the prediction model"""
    global evaluator
    try:
        evaluator = ModelEvaluator(model_path)
        print(f"✅ Model loaded successfully from: {model_path}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


@app.get("/", tags=["General"])
async def root():
    """API root endpoint"""
    return {
        "message": "Bug Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "/predict": "POST - Predict bug risk for a single file",
            "/predict/batch": "POST - Predict bug risk for multiple files",
            "/predict/file": "POST - Predict from uploaded CSV file",
            "/model/info": "GET - Get model information",
            "/model/reload": "POST - Reload the model",
            "/metrics/required": "GET - Get required metrics list",
            "/health": "GET - Check API health"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if evaluator else "degraded",
        model_loaded=evaluator is not None,
        api_version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get model information"""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = None
    if hasattr(evaluator.model, 'feature_names'):
        feature_names = evaluator.model['feature_names']
    elif isinstance(evaluator.model, dict) and 'feature_names' in evaluator.model:
        feature_names = evaluator.model['feature_names']

    return ModelInfo(
        model_path=str(evaluator.model_path),
        model_type=evaluator.model_type,
        status="ready",
        feature_count=len(feature_names) if feature_names else None,
        required_metrics=feature_names
    )


@app.post("/model/reload", tags=["Model"])
async def reload_model(model_path: Optional[str] = None):
    """Reload the model"""
    path = model_path or MODEL_PATH
    if load_model(path):
        return {"message": "Model reloaded successfully", "model_path": path}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")


@app.get("/metrics/required", tags=["Metrics"])
async def get_required_metrics():
    """Get list of required metrics"""
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    feature_names = None
    if hasattr(evaluator.model, 'feature_names'):
        feature_names = evaluator.model['feature_names']
    elif isinstance(evaluator.model, dict) and 'feature_names' in evaluator.model:
        feature_names = evaluator.model['feature_names']

    if feature_names:
        return {
            "required_metrics": feature_names,
            "count": len(feature_names),
            "categories": {
                "size": [m for m in feature_names if any(x in m.upper() for x in ['LOC', 'LLOC', 'CLOC'])],
                "complexity": [m for m in feature_names if any(x in m.upper() for x in ['WMC', 'MCCABE', 'CC'])],
                "coupling": [m for m in feature_names if any(x in m.upper() for x in ['CBO', 'RFC'])],
                "cohesion": [m for m in feature_names if 'LCOM' in m.upper()],
                "inheritance": [m for m in feature_names if any(x in m.upper() for x in ['DIT', 'NOC'])]
            }
        }
    else:
        return {
            "error": "Feature names not available",
            "suggested_metrics": [
                "LOC", "LLOC", "WMC", "CBO", "RFC", "LCOM5",
                "DIT", "NOC", "McCabe", "NOS"
            ]
        }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single_file(request: PredictionRequest):
    """
    Predict bug risk for a single file

    Args:
        request: Prediction request with file name and metrics

    Returns:
        PredictionResponse with bug prediction details
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get feature names
        feature_names = None
        if hasattr(evaluator.model, 'feature_names'):
            feature_names = evaluator.model['feature_names']
        elif isinstance(evaluator.model, dict) and 'feature_names' in evaluator.model:
            feature_names = evaluator.model['feature_names']

        # Make prediction
        result = evaluator.predict_from_metrics(request.metrics, feature_names)

        # Calculate confidence
        confidence = None
        if result['bug_probability'] is not None:
            confidence = abs(result['bug_probability'] - 0.5) * 2

        # Get top risk factors
        top_risk_factors = None
        if isinstance(evaluator.model, dict) and 'feature_importance' in evaluator.model and feature_names:
            importance = evaluator.model['feature_importance']
            top_indices = np.argsort(importance)[::-1][:5]

            risk_factors = []
            for idx in top_indices:
                if idx < len(feature_names):
                    feature = feature_names[idx]
                    if feature in request.metrics:
                        risk_factors.append({
                            'metric': feature,
                            'value': request.metrics[feature],
                            'importance': float(importance[idx])
                        })

            top_risk_factors = risk_factors

        return PredictionResponse(
            file_name=request.file_name,
            prediction=result['prediction'],
            bug_probability=result.get('bug_probability'),
            clean_probability=result.get('clean_probability'),
            risk_level=result['risk_level'],
            confidence=confidence,
            top_risk_factors=top_risk_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch_files(request: BatchPredictionRequest):
    """
    Predict bug risk for multiple files

    Args:
        request: Batch prediction request with multiple files

    Returns:
        BatchPredictionResponse with predictions for all files
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get feature names
        feature_names = None
        if hasattr(evaluator.model, 'feature_names'):
            feature_names = evaluator.model['feature_names']
        elif isinstance(evaluator.model, dict) and 'feature_names' in evaluator.model:
            feature_names = evaluator.model['feature_names']

        results = []

        # Predict for each file
        for file_data in request.files:
            result = evaluator.predict_from_metrics(file_data.metrics, feature_names)

            # Calculate confidence
            confidence = None
            if result['bug_probability'] is not None:
                confidence = abs(result['bug_probability'] - 0.5) * 2

            results.append(PredictionResponse(
                file_name=file_data.file_name,
                prediction=result['prediction'],
                bug_probability=result.get('bug_probability'),
                clean_probability=result.get('clean_probability'),
                risk_level=result['risk_level'],
                confidence=confidence,
                top_risk_factors=None
            ))

        # Sort by bug probability (highest first)
        results.sort(key=lambda x: x.bug_probability or 0, reverse=True)

        # Count high risk files
        high_risk_count = sum(1 for r in results if r.bug_probability and r.bug_probability > 0.6)
        critical_risk_count = sum(1 for r in results if r.bug_probability and r.bug_probability > 0.8)

        return BatchPredictionResponse(
            total_files=len(results),
            high_risk_files=high_risk_count,
            critical_risk_files=critical_risk_count,
            results=results
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file", tags=["Prediction"])
async def predict_from_csv(file: UploadFile = File(...)):
    """
    Predict bug risk from uploaded CSV file

    Args:
        file: CSV file with software metrics

    Returns:
        JSON with predictions for all files
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Get feature columns
        exclude_cols = ['Name', 'Path', 'bug', 'Project', 'Version']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        if not feature_cols:
            raise HTTPException(status_code=400, detail="No valid feature columns found in CSV")

        X = df[feature_cols].fillna(0).values

        # Predict
        y_pred, y_proba = evaluator.predict(X)

        # Prepare results
        results = []
        for i in range(len(df)):
            bug_prob = float(y_proba[i][1]) if y_proba is not None else None

            result = {
                'file_name': df.iloc[i].get('Name', f'File_{i}'),
                'path': df.iloc[i].get('Path', ''),
                'prediction': int(y_pred[i]),
                'bug_probability': bug_prob,
                'risk_level': evaluator._get_risk_level(bug_prob if bug_prob else 0.5),
                'confidence': abs(bug_prob - 0.5) * 2 if bug_prob else None
            }
            results.append(result)

        # Sort by probability
        results.sort(key=lambda x: x.get('bug_probability', 0), reverse=True)

        # Statistics
        high_risk = sum(1 for r in results if r['risk_level'] in ['HIGH', 'CRITICAL'])
        critical_risk = sum(1 for r in results if r['risk_level'] == 'CRITICAL')

        return {
            'total_files': len(results),
            'high_risk_files': high_risk,
            'critical_risk_files': critical_risk,
            'results': results[:100]  # Return top 100
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/file/download", tags=["Prediction"])
async def predict_and_download_csv(file: UploadFile = File(...)):
    """
    Predict bug risk and download results as CSV

    Args:
        file: CSV file with software metrics

    Returns:
        CSV file with predictions
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))

        # Get feature columns
        exclude_cols = ['Name', 'Path', 'bug', 'Project', 'Version']
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].fillna(0).values

        # Predict
        y_pred, y_proba = evaluator.predict(X)

        # Add predictions to dataframe
        df['predicted_bug'] = y_pred
        if y_proba is not None:
            df['bug_probability'] = y_proba[:, 1]
            df['clean_probability'] = y_proba[:, 0]
            df['risk_level'] = df['bug_probability'].apply(evaluator._get_risk_level)
            df['confidence'] = df['bug_probability'].apply(lambda x: abs(x - 0.5) * 2)

        # Sort by probability
        if 'bug_probability' in df.columns:
            df = df.sort_values('bug_probability', ascending=False)

        # Convert to CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=bug_predictions.csv"}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/code", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_code(request: CodeAnalysisRequest):
    """
    Analyze source code and predict bug risk

    Args:
        request: Code analysis request with source code

    Returns:
        PredictionResponse with bug prediction
    """
    if evaluator is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Extract metrics from code
        metrics = analyze_code(request.code, request.language)

        # Get feature names
        feature_names = None
        if hasattr(evaluator.model, 'feature_names'):
            feature_names = evaluator.model['feature_names']
        elif isinstance(evaluator.model, dict) and 'feature_names' in evaluator.model:
            feature_names = evaluator.model['feature_names']

        # Make prediction
        result = evaluator.predict_from_metrics(metrics, feature_names)

        # Calculate confidence
        confidence = None
        if result['bug_probability'] is not None:
            confidence = abs(result['bug_probability'] - 0.5) * 2

        # Get top risk factors
        top_risk_factors = None
        if isinstance(evaluator.model, dict) and 'feature_importance' in evaluator.model and feature_names:
            importance = evaluator.model['feature_importance']
            top_indices = np.argsort(importance)[::-1][:5]

            risk_factors = []
            for idx in top_indices:
                if idx < len(feature_names):
                    feature = feature_names[idx]
                    if feature in metrics:
                        risk_factors.append({
                            'metric': feature,
                            'value': metrics[feature],
                            'importance': float(importance[idx])
                        })

            top_risk_factors = risk_factors

        return PredictionResponse(
            file_name=request.file_name,
            prediction=result['prediction'],
            bug_probability=result.get('bug_probability'),
            clean_probability=result.get('clean_probability'),
            risk_level=result['risk_level'],
            confidence=confidence,
            top_risk_factors=top_risk_factors
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/examples/single", tags=["Examples"])
async def get_single_prediction_example():
    """Get example request for single file prediction"""
    return {
        "example_request": {
            "file_name": "MyClass.java",
            "metrics": {
                "LOC": 150,
                "LLOC": 120,
                "WMC": 12,
                "CBO": 5,
                "RFC": 25,
                "LCOM5": 0.3,
                "DIT": 2,
                "NOC": 0,
                "McCabe": 8,
                "NOS": 50
            }
        },
        "example_response": {
            "file_name": "MyClass.java",
            "prediction": 0,
            "bug_probability": 0.25,
            "clean_probability": 0.75,
            "risk_level": "LOW",
            "confidence": 0.5,
            "top_risk_factors": [
                {"metric": "WMC", "value": 12, "importance": 0.23},
                {"metric": "CBO", "value": 5, "importance": 0.19}
            ]
        }
    }


@app.get("/examples/batch", tags=["Examples"])
async def get_batch_prediction_example():
    """Get example request for batch prediction"""
    return {
        "example_request": {
            "files": [
                {
                    "file_name": "File1.java",
                    "metrics": {
                        "LOC": 200,
                        "WMC": 25,
                        "CBO": 15,
                        "RFC": 50
                    }
                },
                {
                    "file_name": "File2.java",
                    "metrics": {
                        "LOC": 80,
                        "WMC": 5,
                        "CBO": 3,
                        "RFC": 15
                    }
                }
            ]
        }
    }


def main():
    """Run the FastAPI server"""
    import argparse
    import uvicorn

    global MODEL_PATH

    parser = argparse.ArgumentParser(description='Bug Prediction FastAPI Server')
    parser.add_argument('--model', type=str, default="models/random_forest_model.pkl",
                       help='Path to model file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host address')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port number')
    parser.add_argument('--reload', action='store_true',
                       help='Enable auto-reload')

    args = parser.parse_args()
    MODEL_PATH = args.model

    # Run server
    print(f"\n🚀 Starting Bug Prediction FastAPI Server")
    print(f"📍 Server: http://{args.host}:{args.port}")
    print(f"📚 API Docs: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc: http://{args.host}:{args.port}/redoc")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()