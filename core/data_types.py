"""
Data type detection and conversion utilities for Pandas MCP Server
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from core.config import (
    DATE_DETECTION_THRESHOLD,
    NUMERIC_DETECTION_THRESHOLD,
    CATEGORY_UNIQUE_THRESHOLD
)

class DataTypeDetector:
    """Detect and analyze data types in pandas DataFrames"""
    
    # Common date formats to try
    DATE_FORMATS = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]
    
    # Patterns for different data types
    PATTERNS = {
        "email": re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
        "url": re.compile(r'^https?://[^\s]+$'),
        "phone": re.compile(r'^\+?[\d\s\-\(\)]+$'),
        "ip_address": re.compile(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$'),
        "postal_code": re.compile(r'^[A-Z0-9\s\-]{3,10}$', re.IGNORECASE),
        "currency": re.compile(r'^[$€£¥₹]\s*[\d,]+\.?\d*$'),
        "percentage": re.compile(r'^[\d.]+\s*%$'),
    }
    
    @staticmethod
    def detect_column_type(series: pd.Series) -> Dict[str, Any]:
        """
        Detect the data type of a pandas Series
        
        Returns detailed type information including:
        - Primary type (numeric, datetime, categorical, text, etc.)
        - Subtype (integer, float, email, url, etc.)
        - Conversion suggestions
        """
        result = {
            "dtype": str(series.dtype),
            "primary_type": "unknown",
            "subtype": None,
            "nullable": series.isnull().any(),
            "unique_count": series.nunique(),
            "unique_ratio": series.nunique() / len(series) if len(series) > 0 else 0,
            "null_count": series.isnull().sum(),
            "null_ratio": series.isnull().sum() / len(series) if len(series) > 0 else 0,
            "suggestions": []
        }
        
        # Handle empty series
        if len(series) == 0 or series.isnull().all():
            result["primary_type"] = "empty"
            return result
        
        # Get non-null values for analysis
        non_null = series.dropna()
        
        # Check if already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            result["primary_type"] = "datetime"
            result["subtype"] = "datetime64"
            return result
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series):
            result["primary_type"] = "numeric"
            if pd.api.types.is_integer_dtype(series):
                result["subtype"] = "integer"
                # Check if it could be a smaller int type
                if series.min() >= 0 and series.max() <= 255:
                    result["suggestions"].append("Convert to uint8 to save memory")
                elif series.min() >= -32768 and series.max() <= 32767:
                    result["suggestions"].append("Convert to int16 to save memory")
            else:
                result["subtype"] = "float"
                # Check if it could be float32
                if series.min() >= -3.4e38 and series.max() <= 3.4e38:
                    result["suggestions"].append("Convert to float32 to save memory")
            return result
        
        # Check if boolean
        if pd.api.types.is_bool_dtype(series):
            result["primary_type"] = "boolean"
            return result
        
        # For object/string types, do deeper analysis
        if series.dtype == 'object':
            # Try to detect dates
            date_parsed = DataTypeDetector._try_parse_dates(non_null)
            if date_parsed is not None and len(date_parsed) / len(non_null) > DATE_DETECTION_THRESHOLD:
                result["primary_type"] = "datetime"
                result["subtype"] = "date_string"
                result["suggestions"].append("Convert to datetime using pd.to_datetime()")
                return result
            
            # Try to detect numeric strings
            numeric_parsed = DataTypeDetector._try_parse_numeric(non_null)
            if numeric_parsed is not None and len(numeric_parsed) / len(non_null) > NUMERIC_DETECTION_THRESHOLD:
                result["primary_type"] = "numeric"
                result["subtype"] = "numeric_string"
                result["suggestions"].append("Convert to numeric using pd.to_numeric()")
                return result
            
            # Check for specific string patterns
            string_samples = non_null.astype(str).head(100)  # Sample for performance
            pattern_matches = DataTypeDetector._detect_string_patterns(string_samples)
            
            if pattern_matches:
                result["primary_type"] = "text"
                result["subtype"] = pattern_matches[0]  # Primary pattern
                result["patterns_detected"] = pattern_matches
            else:
                # Check if categorical
                if result["unique_ratio"] < CATEGORY_UNIQUE_THRESHOLD:
                    result["primary_type"] = "categorical"
                    result["suggestions"].append("Convert to category dtype to save memory")
                else:
                    result["primary_type"] = "text"
                    result["subtype"] = "freetext"
        
        return result
    
    @staticmethod
    def _try_parse_dates(series: pd.Series) -> Optional[pd.Series]:
        """Try to parse a series as dates"""
        try:
            # First try pandas automatic parsing
            parsed = pd.to_datetime(series, errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except:
            pass
        
        # Try specific formats
        for date_format in DataTypeDetector.DATE_FORMATS:
            try:
                parsed = pd.to_datetime(series, format=date_format, errors='coerce')
                if parsed.notna().sum() / len(series) > 0.5:  # At least 50% parsed
                    return parsed
            except:
                continue
        
        return None
    
    @staticmethod
    def _try_parse_numeric(series: pd.Series) -> Optional[pd.Series]:
        """Try to parse a series as numeric"""
        try:
            # Remove common numeric formatting
            cleaned = series.astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
            parsed = pd.to_numeric(cleaned, errors='coerce')
            if parsed.notna().sum() > 0:
                return parsed
        except:
            pass
        return None
    
    @staticmethod
    def _detect_string_patterns(series: pd.Series) -> List[str]:
        """Detect common string patterns in a series"""
        patterns_found = []
        
        for pattern_name, pattern_regex in DataTypeDetector.PATTERNS.items():
            matches = series.astype(str).apply(lambda x: bool(pattern_regex.match(str(x))))
            if matches.sum() / len(series) > 0.8:  # 80% match threshold
                patterns_found.append(pattern_name)
        
        return patterns_found
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Optimize DataFrame dtypes to reduce memory usage
        
        Args:
            df: DataFrame to optimize
            aggressive: If True, apply more aggressive optimizations
        
        Returns:
            Tuple of (optimized_dataframe, changes_made)
        """
        optimized = df.copy()
        changes = {}
        
        for col in optimized.columns:
            original_dtype = str(optimized[col].dtype)
            
            # Skip if already optimized
            if original_dtype in ['category', 'bool']:
                continue
            
            # Optimize integers
            if optimized[col].dtype in ['int64', 'int32', 'int16']:
                min_val = optimized[col].min()
                max_val = optimized[col].max()
                
                if min_val >= 0:  # Unsigned integers
                    if max_val <= 255:
                        optimized[col] = optimized[col].astype('uint8')
                        changes[col] = f"{original_dtype} -> uint8"
                    elif max_val <= 65535:
                        optimized[col] = optimized[col].astype('uint16')
                        changes[col] = f"{original_dtype} -> uint16"
                    elif max_val <= 4294967295:
                        optimized[col] = optimized[col].astype('uint32')
                        changes[col] = f"{original_dtype} -> uint32"
                else:  # Signed integers
                    if min_val >= -128 and max_val <= 127:
                        optimized[col] = optimized[col].astype('int8')
                        changes[col] = f"{original_dtype} -> int8"
                    elif min_val >= -32768 and max_val <= 32767:
                        optimized[col] = optimized[col].astype('int16')
                        changes[col] = f"{original_dtype} -> int16"
                    elif min_val >= -2147483648 and max_val <= 2147483647:
                        optimized[col] = optimized[col].astype('int32')
                        changes[col] = f"{original_dtype} -> int32"
            
            # Optimize floats
            elif optimized[col].dtype == 'float64':
                if aggressive:
                    optimized[col] = optimized[col].astype('float32')
                    changes[col] = "float64 -> float32"
            
            # Optimize objects/strings
            elif optimized[col].dtype == 'object':
                unique_ratio = optimized[col].nunique() / len(optimized[col])
                if unique_ratio < CATEGORY_UNIQUE_THRESHOLD:
                    optimized[col] = optimized[col].astype('category')
                    changes[col] = "object -> category"
        
        return optimized, changes
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        memory_usage = df.memory_usage(deep=True)
        total_memory = memory_usage.sum()
        
        return {
            "total_mb": total_memory / 1024 / 1024,
            "total_bytes": int(total_memory),
            "by_column": {
                col: {
                    "bytes": int(memory_usage[col]),
                    "mb": memory_usage[col] / 1024 / 1024,
                    "percentage": (memory_usage[col] / total_memory * 100) if total_memory > 0 else 0
                }
                for col in df.columns
            },
            "average_per_row": total_memory / len(df) if len(df) > 0 else 0,
            "shape": df.shape,
            "optimization_potential": DataTypeDetector._estimate_optimization_potential(df)
        }
    
    @staticmethod
    def _estimate_optimization_potential(df: pd.DataFrame) -> Dict[str, Any]:
        """Estimate potential memory savings from optimization"""
        potential_savings = 0
        suggestions = []
        
        for col in df.columns:
            if df[col].dtype == 'float64':
                potential_savings += df[col].memory_usage() * 0.5  # Could save 50% with float32
                suggestions.append(f"{col}: float64 -> float32")
            elif df[col].dtype == 'int64':
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val >= -32768 and max_val <= 32767:
                    potential_savings += df[col].memory_usage() * 0.75  # Could save 75% with int16
                    suggestions.append(f"{col}: int64 -> int16")
            elif df[col].dtype == 'object':
                unique_ratio = df[col].nunique() / len(df[col])
                if unique_ratio < 0.5:
                    # Rough estimate for category savings
                    potential_savings += df[col].memory_usage() * 0.7
                    suggestions.append(f"{col}: object -> category")
        
        current_memory = df.memory_usage(deep=True).sum()
        
        return {
            "current_mb": current_memory / 1024 / 1024,
            "potential_mb": (current_memory - potential_savings) / 1024 / 1024,
            "savings_mb": potential_savings / 1024 / 1024,
            "savings_percentage": (potential_savings / current_memory * 100) if current_memory > 0 else 0,
            "suggestions": suggestions[:5]  # Top 5 suggestions
        }

def analyze_data_types(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data type analysis for a DataFrame
    
    Returns a complete type analysis including:
    - Type distribution
    - Memory usage
    - Optimization suggestions
    """
    detector = DataTypeDetector()
    
    type_analysis = {
        "columns": {},
        "summary": {
            "total_columns": len(df.columns),
            "numeric_columns": 0,
            "datetime_columns": 0,
            "categorical_columns": 0,
            "text_columns": 0,
            "boolean_columns": 0,
        },
        "memory": detector.get_memory_usage(df),
        "optimization": {}
    }
    
    # Analyze each column
    for col in df.columns:
        col_analysis = detector.detect_column_type(df[col])
        type_analysis["columns"][col] = col_analysis
        
        # Update summary counts
        primary_type = col_analysis["primary_type"]
        if primary_type == "numeric":
            type_analysis["summary"]["numeric_columns"] += 1
        elif primary_type == "datetime":
            type_analysis["summary"]["datetime_columns"] += 1
        elif primary_type == "categorical":
            type_analysis["summary"]["categorical_columns"] += 1
        elif primary_type == "text":
            type_analysis["summary"]["text_columns"] += 1
        elif primary_type == "boolean":
            type_analysis["summary"]["boolean_columns"] += 1
    
    # Try optimization
    optimized_df, changes = detector.optimize_dtypes(df)
    if changes:
        type_analysis["optimization"]["suggested_changes"] = changes
        type_analysis["optimization"]["memory_before_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
        type_analysis["optimization"]["memory_after_mb"] = optimized_df.memory_usage(deep=True).sum() / 1024 / 1024
        type_analysis["optimization"]["savings_percentage"] = (
            (1 - optimized_df.memory_usage(deep=True).sum() / df.memory_usage(deep=True).sum()) * 100
        )
    
    return type_analysis