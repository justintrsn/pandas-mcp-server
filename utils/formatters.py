"""
Output formatting utilities.
Handles formatting of pandas execution results for MCP responses.
"""

from typing import Any, Dict, Union
import pandas as pd
import numpy as np
import json


def format_execution_result(result: Any, result_type: str) -> Any:
    """
    Format execution result based on its type.
    
    Args:
        result: The raw execution result
        result_type: Type of the result (dataframe, series, list, etc.)
        
    Returns:
        Formatted result suitable for JSON serialization
    """
    if result_type == "dataframe":
        return format_dataframe(result)
    elif result_type == "series":
        return format_series(result)
    elif result_type == "list":
        return format_list(result)
    elif result_type == "dict":
        return format_dict(result)
    elif result_type == "scalar":
        return format_scalar(result)
    else:
        return str(result)


def format_dataframe(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
    """
    Format a DataFrame for response.
    
    Args:
        df: DataFrame to format
        max_rows: Maximum rows to include in data
        
    Returns:
        Formatted DataFrame dictionary
    """
    if not isinstance(df, pd.DataFrame):
        return {"error": "Not a DataFrame"}
    
    return {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview": df.head(10).to_string(max_cols=10),
        "data": df.head(max_rows).to_dict('records'),
        "truncated": len(df) > max_rows
    }


def format_series(series: pd.Series, max_items: int = 100) -> Dict[str, Any]:
    """
    Format a Series for response.
    
    Args:
        series: Series to format
        max_items: Maximum items to include
        
    Returns:
        Formatted Series dictionary
    """
    if not isinstance(series, pd.Series):
        return {"error": "Not a Series"}
    
    return {
        "name": series.name,
        "dtype": str(series.dtype),
        "length": len(series),
        "preview": series.head(10).to_string(),
        "data": series.head(max_items).tolist(),
        "index": series.head(max_items).index.tolist(),
        "truncated": len(series) > max_items
    }


def format_list(lst: list, max_items: int = 1000) -> Union[list, Dict[str, Any]]:
    """
    Format a list for response.
    
    Args:
        lst: List to format
        max_items: Maximum items to include
        
    Returns:
        Formatted list or truncated version with metadata
    """
    if len(lst) <= max_items:
        return lst
    else:
        return {
            "data": lst[:max_items],
            "total_length": len(lst),
            "truncated": True
        }


def format_dict(dct: dict) -> dict:
    """
    Format a dictionary for response.
    
    Args:
        dct: Dictionary to format
        
    Returns:
        JSON-serializable dictionary
    """
    try:
        # Try to JSON serialize to ensure it's safe
        json.dumps(dct)
        return dct
    except (TypeError, ValueError):
        # Convert non-serializable values to strings
        return {
            str(k): str(v) if not isinstance(v, (str, int, float, bool, list, dict, type(None))) else v
            for k, v in dct.items()
        }


def format_scalar(value: Any) -> Any:
    """
    Format a scalar value for response.
    
    Args:
        value: Scalar value to format
        
    Returns:
        JSON-serializable value
    """
    # Handle numpy types
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return str(value)


def format_error(error: Exception) -> Dict[str, str]:
    """
    Format an error for response.
    
    Args:
        error: Exception to format
        
    Returns:
        Error dictionary
    """
    return {
        "error": str(error),
        "type": type(error).__name__
    }