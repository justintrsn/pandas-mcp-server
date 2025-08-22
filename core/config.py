"""
Configuration and constants for Pandas MCP Server
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server Configuration
SERVER_NAME = "pandas-mcp-server"
SERVER_VERSION = "0.1.0"
SERVER_DESCRIPTION = "A powerful MCP server for pandas data analysis with Chart.js visualizations"

# Network Configuration
SERVER_HOST = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("MCP_SERVER_PORT", "8000"))
SERVER_TRANSPORT = os.getenv("MCP_SERVER_TRANSPORT", "sse")

# Logging Configuration
LOG_LEVEL = os.getenv("MCP_LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("MCP_LOG_FILE", "logs/pandas_mcp.log")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data Limits
MAX_DATAFRAME_SIZE_MB = int(os.getenv("MCP_MAX_DATAFRAME_SIZE_MB", "100"))
MAX_FILE_SIZE_MB = int(os.getenv("MCP_MAX_FILE_SIZE_MB", "100"))
MAX_DATAFRAMES = int(os.getenv("MCP_MAX_DATAFRAMES", "20"))
MAX_ROWS_PREVIEW = int(os.getenv("MCP_MAX_ROWS_PREVIEW", "100"))
MAX_COLUMNS_DISPLAY = int(os.getenv("MCP_MAX_COLUMNS_DISPLAY", "50"))
CHUNK_SIZE = int(os.getenv("MCP_CHUNK_SIZE", "10000"))

# Session Configuration
SESSION_TIMEOUT_MINUTES = int(os.getenv("MCP_SESSION_TIMEOUT_MINUTES", "60"))
ENABLE_CACHING = os.getenv("MCP_ENABLE_CACHING", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("MCP_CACHE_TTL_SECONDS", "300"))

# Security Configuration
ENABLE_AUTH = os.getenv("MCP_ENABLE_AUTH", "false").lower() == "true"
API_KEY = os.getenv("MCP_API_KEY", None)
ALLOWED_FILE_EXTENSIONS = os.getenv(
    "MCP_ALLOWED_FILE_EXTENSIONS", 
    "csv,xlsx,xls,json,parquet,tsv,txt"
).split(",")

# Forbidden operations for code execution
FORBIDDEN_OPERATIONS = [
    "exec", "eval", "__import__", "compile", "open",
    "file", "input", "raw_input", "globals", "locals",
    "vars", "dir", "getattr", "setattr", "delattr",
    "reload", "__builtins__", "subprocess", "os.",
    "sys.", "importlib", "socket", "urllib", "requests"
]

# Safe pandas operations whitelist
SAFE_PANDAS_OPERATIONS = [
    # Data loading
    "read_csv", "read_excel", "read_json", "read_parquet",
    "read_clipboard", "read_sql", "read_html",
    
    # Data inspection
    "head", "tail", "info", "describe", "shape", "columns",
    "dtypes", "index", "values", "empty", "size", "ndim",
    
    # Data selection
    "loc", "iloc", "at", "iat", "query", "filter", "where",
    "mask", "take", "sample", "nlargest", "nsmallest",
    
    # Data manipulation
    "drop", "dropna", "fillna", "replace", "rename",
    "reindex", "reset_index", "set_index", "sort_values",
    "sort_index", "pivot", "pivot_table", "melt", "stack",
    "unstack", "transpose", "T", "swapaxes", "assign",
    
    # Aggregation
    "groupby", "agg", "aggregate", "sum", "mean", "median",
    "mode", "std", "var", "min", "max", "count", "nunique",
    "value_counts", "unique", "cumsum", "cumprod", "cummax",
    "cummin", "rolling", "expanding", "ewm",
    
    # Merging
    "merge", "join", "concat", "append", "combine",
    "combine_first", "update",
    
    # Time series
    "resample", "asfreq", "shift", "diff", "pct_change",
    "to_datetime", "to_timedelta", "to_period",
    
    # String operations
    "str", "contains", "startswith", "endswith", "match",
    "extract", "extractall", "len", "lower", "upper",
    
    # Statistical
    "corr", "cov", "rank", "quantile", "clip", "round",
    "abs", "all", "any", "interpolate",
    
    # Type conversion
    "astype", "to_numeric", "to_datetime", "to_timedelta",
    "convert_dtypes", "infer_objects",
    
    # Export
    "to_csv", "to_excel", "to_json", "to_parquet", "to_dict",
    "to_list", "to_numpy", "to_string", "to_markdown", "to_html"
]

# File Paths
BASE_DIR = Path(__file__).parent.parent
CORE_DIR = BASE_DIR / "core"
CHARTS_DIR = BASE_DIR / "charts"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = CORE_DIR / "chart_generators" / "templates"

# Create directories if they don't exist
for directory in [CHARTS_DIR, LOGS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Chart Configuration
CHART_TYPES = ["bar", "line", "pie", "scatter", "area", "doughnut", "radar", "polar"]
CHART_DEFAULT_WIDTH = 800
CHART_DEFAULT_HEIGHT = 600
CHART_COLOR_PALETTE = [
    "#FF6384", "#36A2EB", "#FFCE56", "#4BC0C0", "#9966FF",
    "#FF9F40", "#FF6384", "#C9CBCF", "#4BC0C0", "#FF6384"
]

# Data Type Detection Thresholds
DATE_DETECTION_THRESHOLD = 0.8  # 80% of non-null values must parse as dates
NUMERIC_DETECTION_THRESHOLD = 0.9  # 90% must be numeric
CATEGORY_UNIQUE_THRESHOLD = 0.5  # Less than 50% unique values for categorical

# Performance Configuration
MAX_WORKERS = int(os.getenv("MCP_MAX_WORKERS", "4"))
ASYNC_TIMEOUT = int(os.getenv("MCP_ASYNC_TIMEOUT", "30"))

# CORS Configuration (for SSE)
CORS_ORIGINS = os.getenv(
    "MCP_CORS_ORIGINS",
    '["http://localhost:3000", "https://claude.ai", "http://127.0.0.1:*"]'
)

# Error Messages
ERROR_MESSAGES = {
    "file_not_found": "File not found: {filepath}",
    "file_too_large": "File size ({size}MB) exceeds maximum allowed size ({max_size}MB)",
    "invalid_file_type": "Invalid file type. Allowed types: {allowed_types}",
    "dataframe_not_found": "DataFrame '{name}' not found in session",
    "invalid_code": "Invalid or unsafe code detected",
    "execution_error": "Code execution failed: {error}",
    "memory_limit": "Operation would exceed memory limit",
    "timeout": "Operation timed out after {timeout} seconds",
    "invalid_chart_type": "Invalid chart type. Supported types: {supported_types}",
}

# Success Messages
SUCCESS_MESSAGES = {
    "file_loaded": "Successfully loaded {filepath} ({rows} rows, {cols} columns)",
    "code_executed": "Code executed successfully",
    "chart_generated": "Chart generated successfully: {filepath}",
    "dataframe_saved": "DataFrame '{name}' saved successfully",
}

# Metadata Output Configuration
METADATA_SECTIONS = {
    "basic_info": True,
    "column_analysis": True,
    "data_quality": True,
    "statistics": True,
    "relationships": True,
    "recommendations": True,
    "memory_usage": True,
    "sample_data": True,
}

def get_config() -> Dict[str, Any]:
    """Get all configuration as a dictionary"""
    return {
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "transport": SERVER_TRANSPORT,
        },
        "limits": {
            "max_dataframe_size_mb": MAX_DATAFRAME_SIZE_MB,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_dataframes": MAX_DATAFRAMES,
            "max_rows_preview": MAX_ROWS_PREVIEW,
        },
        "security": {
            "enable_auth": ENABLE_AUTH,
            "forbidden_operations": FORBIDDEN_OPERATIONS,
            "allowed_extensions": ALLOWED_FILE_EXTENSIONS,
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "charts_dir": str(CHARTS_DIR),
            "logs_dir": str(LOGS_DIR),
            "data_dir": str(DATA_DIR),
        }
    }

def validate_config():
    """Validate configuration on startup"""
    errors = []
    
    if MAX_DATAFRAME_SIZE_MB < 1:
        errors.append("MAX_DATAFRAME_SIZE_MB must be at least 1")
    
    if SERVER_PORT < 1 or SERVER_PORT > 65535:
        errors.append("SERVER_PORT must be between 1 and 65535")
    
    if not ALLOWED_FILE_EXTENSIONS:
        errors.append("At least one file extension must be allowed")
    
    if ENABLE_AUTH and not API_KEY:
        errors.append("API_KEY must be set when authentication is enabled")
    
    if errors:
        raise ValueError(f"Configuration errors: {'; '.join(errors)}")

# Validate on import
validate_config()