"""
Input validation utilities for Pandas MCP Server.
Validates user inputs, file paths, and parameters.
"""

import re
import os
from pathlib import Path
from typing import Tuple, Optional, List, Any
from core.config import (
    ALLOWED_FILE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    MAX_DATAFRAME_SIZE_MB,
    CHUNK_SIZE
)


class InputValidator:
    """Validates various types of user inputs"""
    
    @staticmethod
    def validate_filepath(filepath: str) -> Tuple[bool, str]:
        """
        Validate a file path for safety and existence.
        
        Args:
            filepath: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message or cleaned_path)
        """
        if not filepath:
            return False, "Empty filepath provided"
        
        # Remove any null bytes
        filepath = filepath.replace('\x00', '')
        
        # Normalize the path first to resolve any . or .. components
        normalized = os.path.normpath(filepath)
        
        # Check if the normalized path tries to escape current directory
        # by going up with .. or by being absolute
        if normalized.startswith('..') or os.path.isabs(normalized):
            return False, "Path traversal detected"
        
        # Additional security checks for suspicious patterns
        suspicious_patterns = [
            '../',  # Parent directory traversal
            '..\\',  # Parent directory traversal (Windows)
            '\\\\',  # UNC path
        ]
        
        for pattern in suspicious_patterns:
            if pattern in filepath:
                return False, "Path traversal detected"
        
        # Check file extension
        path = Path(normalized)
        extension = path.suffix.lower().replace('.', '')
        
        if extension not in ALLOWED_FILE_EXTENSIONS:
            return False, f"File type '{extension}' not allowed. Supported: {', '.join(ALLOWED_FILE_EXTENSIONS)}"
        
        # Check if file exists
        if not path.exists():
            return False, f"File not found: {normalized}"
        
        # Check file size
        file_size_mb = path.stat().st_size / 1024 / 1024
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, f"File too large: {file_size_mb:.2f}MB > {MAX_FILE_SIZE_MB}MB"
        
        return True, normalized

    @staticmethod
    def validate_dataframe_name(name: str) -> Tuple[bool, str]:
        """
        Validate a DataFrame name.
        
        Args:
            name: DataFrame name to validate
            
        Returns:
            Tuple of (is_valid, error_message or cleaned_name)
        """
        if not name:
            return False, "DataFrame name cannot be empty"
        
        # Check length
        if len(name) > 100:
            return False, "DataFrame name too long (max 100 characters)"
        
        # Check for valid Python identifier (with some flexibility)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', name):
            return False, "Invalid DataFrame name. Must start with letter and contain only letters, numbers, and underscores"
        
        # Check for reserved names
        reserved = ['df', 'pd', 'np', 'result', 'self', 'True', 'False', 'None']
        if name in reserved:
            return True, name  # Actually allow 'df' as it's common
        
        return True, name
    
    @staticmethod
    def validate_session_id(session_id: str) -> Tuple[bool, str]:
        """
        Validate a session ID.
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Tuple of (is_valid, error_message or cleaned_id)
        """
        if not session_id:
            return True, "default"  # Use default if empty
        
        # Check length
        if len(session_id) > 50:
            return False, "Session ID too long (max 50 characters)"
        
        # Allow alphanumeric, dash, underscore
        if not re.match(r'^[a-zA-Z0-9_-]+$', session_id):
            return False, "Invalid session ID. Use only letters, numbers, dash, and underscore"
        
        return True, session_id
    
    @staticmethod
    def validate_sample_size(sample_size: Any) -> Tuple[bool, int]:
        """
        Validate sample size parameter.
        
        Args:
            sample_size: Sample size to validate
            
        Returns:
            Tuple of (is_valid, error_message or validated_size)
        """
        try:
            size = int(sample_size)
            
            if size < 1:
                return False, "Sample size must be at least 1"
            
            if size > 1000000:
                return False, "Sample size too large (max 1,000,000)"
            
            return True, size
            
        except (TypeError, ValueError):
            return False, "Sample size must be an integer"
    
    @staticmethod
    def validate_delimiter(delimiter: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate CSV delimiter.
        
        Args:
            delimiter: Delimiter to validate
            
        Returns:
            Tuple of (is_valid, error_message or delimiter)
        """
        if delimiter is None:
            return True, None
        
        if not isinstance(delimiter, str):
            return False, "Delimiter must be a string"
        
        if len(delimiter) > 5:
            return False, "Delimiter too long (max 5 characters)"
        
        # Common delimiters
        valid_delimiters = [',', '\t', ';', '|', ' ', ':', '~']
        if len(delimiter) == 1 and delimiter not in valid_delimiters:
            # Still allow but warn
            pass
        
        return True, delimiter
    
    @staticmethod
    def validate_encoding(encoding: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate file encoding.
        
        Args:
            encoding: Encoding to validate
            
        Returns:
            Tuple of (is_valid, error_message or encoding)
        """
        if encoding is None:
            return True, None
        
        valid_encodings = [
            'utf-8', 'utf-16', 'utf-32',
            'latin-1', 'iso-8859-1',
            'cp1252', 'windows-1252',
            'ascii', 'big5', 'gb2312', 'gbk',
            'shift_jis', 'euc-jp', 'euc-kr'
        ]
        
        if encoding.lower() not in valid_encodings:
            return False, f"Unsupported encoding. Try: {', '.join(valid_encodings[:5])}"
        
        return True, encoding.lower()
    
    @staticmethod
    def validate_sheet_name(sheet_name: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate Excel sheet name.
        
        Args:
            sheet_name: Sheet name to validate
            
        Returns:
            Tuple of (is_valid, error_message or sheet_name)
        """
        if sheet_name is None:
            return True, None
        
        if not isinstance(sheet_name, str):
            # Could be an integer for sheet index
            try:
                index = int(sheet_name)
                if index < 0:
                    return False, "Sheet index must be non-negative"
                if index > 100:
                    return False, "Sheet index too large (max 100)"
                return True, index
            except (TypeError, ValueError):
                return False, "Sheet name must be string or integer"
        
        if len(sheet_name) > 100:
            return False, "Sheet name too long (max 100 characters)"
        
        return True, sheet_name
    
    @staticmethod
    def validate_column_names(columns: List[str]) -> Tuple[bool, str]:
        """
        Validate column names list.
        
        Args:
            columns: List of column names
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not columns:
            return False, "No columns provided"
        
        if not isinstance(columns, list):
            return False, "Columns must be a list"
        
        if len(columns) > 1000:
            return False, "Too many columns (max 1000)"
        
        # Check for duplicates
        if len(columns) != len(set(columns)):
            duplicates = [c for c in columns if columns.count(c) > 1]
            return False, f"Duplicate column names: {duplicates[:5]}"
        
        # Check each column name
        for col in columns:
            if not isinstance(col, str):
                return False, f"Column name must be string, got {type(col)}"
            if len(col) > 200:
                return False, f"Column name too long: {col[:50]}..."
        
        return True, "Valid"


class ParameterValidator:
    """Validates function parameters and options"""
    
    @staticmethod
    def validate_pandas_code(code: str) -> Tuple[bool, str]:
        """
        Basic validation of pandas code (security check done elsewhere).
        
        Args:
            code: Pandas code to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not code:
            return False, "Empty code provided"
        
        if not isinstance(code, str):
            return False, "Code must be a string"
        
        # Check length
        if len(code) > 10000:
            return False, "Code too long (max 10000 characters)"
        
        # Check for basic syntax issues
        if code.count('(') != code.count(')'):
            return False, "Unbalanced parentheses"
        
        if code.count('[') != code.count(']'):
            return False, "Unbalanced brackets"
        
        if code.count('{') != code.count('}'):
            return False, "Unbalanced braces"
        
        return True, "Valid"
    
    @staticmethod
    def validate_memory_usage(size_mb: float) -> Tuple[bool, str]:
        """
        Validate if operation would exceed memory limits.
        
        Args:
            size_mb: Size in megabytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if size_mb > MAX_DATAFRAME_SIZE_MB:
            return False, f"Would exceed memory limit: {size_mb:.2f}MB > {MAX_DATAFRAME_SIZE_MB}MB"
        
        # Check system memory
        try:
            import psutil
            available_mb = psutil.virtual_memory().available / 1024 / 1024
            if size_mb > available_mb * 0.8:  # Use max 80% of available
                return False, f"Insufficient system memory: need {size_mb:.2f}MB, available {available_mb:.2f}MB"
        except ImportError:
            pass  # psutil not available, skip system check
        
        return True, "Valid"


def validate_tool_inputs(tool_name: str, **kwargs) -> Tuple[bool, str]:
    """
    Validate inputs for a specific tool.
    
    Args:
        tool_name: Name of the tool
        **kwargs: Tool parameters
        
    Returns:
        Tuple of (all_valid, error_message)
    """
    validator = InputValidator()
    param_validator = ParameterValidator()
    
    if tool_name == "read_metadata_tool":
        # Validate filepath
        if 'filepath' in kwargs:
            valid, msg = validator.validate_filepath(kwargs['filepath'])
            if not valid:
                return False, msg
        
        # Validate sample_size
        if 'sample_size' in kwargs:
            valid, msg = validator.validate_sample_size(kwargs['sample_size'])
            if not valid:
                return False, msg
    
    elif tool_name == "run_pandas_code_tool":
        # Validate code
        if 'code' in kwargs:
            valid, msg = param_validator.validate_pandas_code(kwargs['code'])
            if not valid:
                return False, msg
        
        # Validate target_df
        if 'target_df' in kwargs:
            valid, msg = validator.validate_dataframe_name(kwargs['target_df'])
            if not valid:
                return False, msg
    
    elif tool_name == "load_dataframe_tool":
        # Validate filepath
        if 'filepath' in kwargs:
            valid, msg = validator.validate_filepath(kwargs['filepath'])
            if not valid:
                return False, msg
        
        # Validate df_name
        if 'df_name' in kwargs:
            valid, msg = validator.validate_dataframe_name(kwargs['df_name'])
            if not valid:
                return False, msg
        
        # Validate optional parameters
        if 'delimiter' in kwargs:
            valid, msg = validator.validate_delimiter(kwargs['delimiter'])
            if not valid:
                return False, msg
        
        if 'encoding' in kwargs:
            valid, msg = validator.validate_encoding(kwargs['encoding'])
            if not valid:
                return False, msg
        
        if 'sheet_name' in kwargs:
            valid, msg = validator.validate_sheet_name(kwargs['sheet_name'])
            if not valid:
                return False, msg
    
    # Validate session_id for all tools that use it
    if 'session_id' in kwargs:
        valid, msg = validator.validate_session_id(kwargs['session_id'])
        if not valid:
            return False, msg
    
    return True, "All inputs valid"