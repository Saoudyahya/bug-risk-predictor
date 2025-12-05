"""
Code analysis endpoint for extracting metrics from source code
"""

import re
from typing import Dict
from fastapi import HTTPException
from pydantic import BaseModel


class CodeAnalysisRequest(BaseModel):
    """Request for code analysis"""
    code: str
    language: str = "java"
    file_name: str = "Code.java"


class CodeMetrics(BaseModel):
    """Extracted code metrics"""
    LOC: int = 0
    LLOC: int = 0
    NOS: int = 0
    WMC: int = 0
    McCabe: int = 0
    CBO: int = 0
    RFC: int = 0
    LCOM5: float = 0.0
    DIT: int = 0
    NOC: int = 0


def analyze_java_code(code: str) -> Dict:
    """
    Extract basic metrics from Java code
    This is a simplified analysis - for production, use proper static analysis tools
    """
    lines = code.split('\n')
    
    # Basic counts
    loc = len(lines)
    lloc = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
    
    # Count statements (simplified)
    statements = len(re.findall(r';', code))
    
    # Count methods
    methods = len(re.findall(r'(public|private|protected)?\s*(static)?\s*\w+\s+\w+\s*\([^)]*\)\s*\{', code))
    
    # Count classes referenced (imports + new statements)
    imports = len(re.findall(r'import\s+[\w.]+;', code))
    news = len(re.findall(r'new\s+\w+', code))
    cbo = imports + (news // 2)  # Approximate coupling
    
    # Cyclomatic complexity (simplified - count decision points)
    decision_points = (
        len(re.findall(r'\bif\b', code)) +
        len(re.findall(r'\bfor\b', code)) +
        len(re.findall(r'\bwhile\b', code)) +
        len(re.findall(r'\bcase\b', code)) +
        len(re.findall(r'\bcatch\b', code))
    )
    mccabe = decision_points + 1
    
    # Response for class (approximate)
    rfc = methods + cbo
    
    # Inheritance depth (check extends keyword)
    dit = 1 if 'extends' in code else 0
    
    # Number of children (can't determine from single file)
    noc = 0
    
    # Lack of cohesion (simplified)
    if methods > 0:
        # Count field accesses in methods
        fields = len(re.findall(r'this\.\w+', code))
        lcom5 = max(0, 1 - (fields / (methods * 2))) if methods > 1 else 0.0
    else:
        lcom5 = 0.0
    
    return {
        'LOC': loc,
        'LLOC': lloc,
        'NOS': statements,
        'WMC': methods,
        'McCabe': mccabe,
        'CBO': min(cbo, 30),  # Cap at reasonable value
        'RFC': min(rfc, 100),
        'LCOM5': round(lcom5, 2),
        'DIT': dit,
        'NOC': noc,
    }


def analyze_python_code(code: str) -> Dict:
    """
    Extract basic metrics from Python code
    """
    lines = code.split('\n')
    
    # Basic counts
    loc = len(lines)
    lloc = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    
    # Count statements
    statements = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
    
    # Count functions/methods
    methods = len(re.findall(r'def\s+\w+\s*\(', code))
    
    # Count imports
    imports = len(re.findall(r'^import\s+|^from\s+', code, re.MULTILINE))
    
    # Cyclomatic complexity
    decision_points = (
        len(re.findall(r'\bif\b', code)) +
        len(re.findall(r'\bfor\b', code)) +
        len(re.findall(r'\bwhile\b', code)) +
        len(re.findall(r'\belif\b', code)) +
        len(re.findall(r'\bexcept\b', code))
    )
    mccabe = decision_points + 1
    
    # Classes
    classes = len(re.findall(r'class\s+\w+', code))
    cbo = imports + classes
    
    return {
        'LOC': loc,
        'LLOC': lloc,
        'NOS': statements,
        'WMC': methods,
        'McCabe': mccabe,
        'CBO': min(cbo, 30),
        'RFC': methods + imports,
        'LCOM5': 0.3,  # Default value
        'DIT': 1 if 'class' in code and '(' in code else 0,
        'NOC': 0,
    }


def analyze_code(code: str, language: str = "java") -> Dict:
    """
    Analyze code and extract metrics based on language
    """
    if not code or len(code.strip()) < 10:
        raise HTTPException(status_code=400, detail="Code is too short or empty")
    
    if language.lower() == "java":
        return analyze_java_code(code)
    elif language.lower() == "python":
        return analyze_python_code(code)
    else:
        # Default to basic analysis
        lines = code.split('\n')
        return {
            'LOC': len(lines),
            'LLOC': len([l for l in lines if l.strip()]),
            'NOS': len([l for l in lines if l.strip()]),
            'WMC': 5,
            'McCabe': 3,
            'CBO': 2,
            'RFC': 10,
            'LCOM5': 0.3,
            'DIT': 0,
            'NOC': 0,
        }
