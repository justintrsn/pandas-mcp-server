"""
Data loading module.
Handles loading various file formats into pandas DataFrames.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import pandas as pd

from storage.dataframe_manager import get_manager
from utils.validators import InputValidator, ParameterValidator
from utils.formatters import format_dataframe

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads data files into pandas DataFrames with format detection.
    """
    
    def __init__(self):
        self.df_manager = get_manager()
        self.input_validator = InputValidator()
        self.param_validator = ParameterValidator()
        
        # Supported file extensions and their loaders
        self.loaders = {
            '.csv': self._load_csv,
            '.tsv': self._load_tsv,
            '.txt': self._load_csv,  # Treat as CSV
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.json': self._load_json,
            '.parquet': self._load_parquet
        }
    
    def load(
        self,
        filepath: str,
        df_name: str = "df",
        session_id: str = "default",
        **options
    ) -> Dict[str, Any]:
        """
        Load a file into a DataFrame.
        
        Args:
            filepath: Path to the file
            df_name: Name for the DataFrame
            session_id: Session identifier
            **options: Additional loader-specific options
            
        Returns:
            Success status and DataFrame information
        """
        # VALIDATION happens HERE - not in tools.py
        logger.info(f"Loading file '{filepath}' as DataFrame '{df_name}' in session '{session_id}'")
        
        # 1. Validate filepath using InputValidator
        is_valid, clean_filepath = self.input_validator.validate_filepath(filepath)
        if not is_valid:
            logger.error(f"Invalid filepath: {clean_filepath}")
            return {"success": False, "error": f"Invalid filepath: {clean_filepath}"}
        
        # 2. Validate DataFrame name using InputValidator
        is_valid, clean_df_name = self.input_validator.validate_dataframe_name(df_name)
        if not is_valid:
            logger.error(f"Invalid DataFrame name: {clean_df_name}")
            return {"success": False, "error": f"Invalid DataFrame name: {clean_df_name}"}
        
        # 3. Validate session ID using InputValidator
        is_valid, clean_session_id = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            logger.error(f"Invalid session ID: {clean_session_id}")
            return {"success": False, "error": f"Invalid session ID: {clean_session_id}"}
        
        # 4. Validate and clean options
        cleaned_options = {}
        validation_result = self._validate_and_clean_options(options)
        if not validation_result["success"]:
            return validation_result
        cleaned_options = validation_result["options"]
        
        filepath = Path(clean_filepath)
        
        # Get file extension
        file_ext = filepath.suffix.lower()
        
        # Check if supported
        if file_ext not in self.loaders:
            error_msg = (f"Unsupported file type: {file_ext}. "
                        f"Supported types: {', '.join(self.loaders.keys())}")
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
        
        try:
            logger.info(f"Using {file_ext} loader with options: {cleaned_options}")
            
            # Load using appropriate loader
            df = self.loaders[file_ext](filepath, **cleaned_options)
            
            # Validate memory usage
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            is_valid, memory_msg = self.param_validator.validate_memory_usage(memory_mb)
            if not is_valid:
                logger.error(f"Memory validation failed: {memory_msg}")
                return {"success": False, "error": memory_msg}
            
            logger.info(f"Loaded DataFrame: {df.shape}, Memory: {memory_mb:.2f}MB")
            
            # Add to DataFrame manager
            success, message = self.df_manager.add_dataframe(df, clean_df_name, clean_session_id)
            
            if success:
                logger.info(f"Successfully stored DataFrame '{clean_df_name}': {message}")
                
                # Return comprehensive information
                return {
                    "success": True,
                    "message": message,
                    "dataframe_info": {
                        "name": clean_df_name,
                        "session_id": clean_session_id,
                        "shape": df.shape,
                        "columns": df.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                        "memory_mb": round(memory_mb, 2),
                        "file_info": {
                            "path": str(filepath),
                            "size_mb": round(filepath.stat().st_size / 1024 / 1024, 2),
                            "type": file_ext.replace('.', '')
                        }
                    },
                    "preview": df.head(5).to_string(max_cols=10),
                    "options_used": cleaned_options
                }
            else:
                logger.error(f"Failed to store DataFrame: {message}")
                return {"success": False, "error": message}
                
        except Exception as e:
            error_msg = f"Failed to load {filepath}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {"success": False, "error": error_msg}
    
    def _validate_and_clean_options(self, options: dict) -> Dict[str, Any]:
        """
        Validate and clean loading options.
        
        Args:
            options: Raw options dictionary
            
        Returns:
            Validation result with cleaned options
        """
        cleaned_options = {}
        
        # Validate delimiter
        if 'delimiter' in options:
            is_valid, clean_delimiter = self.input_validator.validate_delimiter(options['delimiter'])
            if not is_valid:
                return {"success": False, "error": f"Invalid delimiter: {clean_delimiter}"}
            if clean_delimiter is not None:
                cleaned_options['delimiter'] = clean_delimiter
        
        # Validate encoding
        if 'encoding' in options:
            is_valid, clean_encoding = self.input_validator.validate_encoding(options['encoding'])
            if not is_valid:
                return {"success": False, "error": f"Invalid encoding: {clean_encoding}"}
            if clean_encoding is not None:
                cleaned_options['encoding'] = clean_encoding
        
        # Validate sheet_name (for Excel files)
        if 'sheet_name' in options:
            is_valid, clean_sheet = self.input_validator.validate_sheet_name(options['sheet_name'])
            if not is_valid:
                return {"success": False, "error": f"Invalid sheet name: {clean_sheet}"}
            if clean_sheet is not None:
                cleaned_options['sheet_name'] = clean_sheet
        
        # Validate other common options
        safe_options = [
            'sep', 'header', 'names', 'index_col', 'usecols', 'dtype',
            'skiprows', 'nrows', 'na_values', 'keep_default_na',
            'na_filter', 'skip_blank_lines', 'parse_dates', 'date_parser',
            'dayfirst', 'thousands', 'decimal', 'lineterminator',
            'quotechar', 'quoting', 'skipinitialspace', 'escapechar',
            'comment', 'compression', 'mangle_dupe_cols', 'orient',
            'lines', 'chunksize', 'iterator'
        ]
        
        for key, value in options.items():
            if key in safe_options and key not in cleaned_options:
                # Basic validation for common types
                if key in ['nrows', 'skiprows'] and value is not None:
                    try:
                        int_value = int(value)
                        if int_value < 0:
                            return {"success": False, "error": f"Invalid {key}: must be non-negative"}
                        cleaned_options[key] = int_value
                    except (ValueError, TypeError):
                        return {"success": False, "error": f"Invalid {key}: must be an integer"}
                
                elif key in ['header'] and value is not None:
                    if value == 'infer' or value is None:
                        cleaned_options[key] = value
                    else:
                        try:
                            header_value = int(value) if isinstance(value, str) and value.isdigit() else value
                            if isinstance(header_value, int) and header_value < 0:
                                return {"success": False, "error": "Invalid header: must be non-negative"}
                            cleaned_options[key] = header_value
                        except (ValueError, TypeError):
                            return {"success": False, "error": "Invalid header parameter"}
                
                elif key in ['dayfirst', 'keep_default_na', 'na_filter', 'skip_blank_lines', 'skipinitialspace', 'mangle_dupe_cols']:
                    # Boolean options
                    if isinstance(value, bool):
                        cleaned_options[key] = value
                    elif isinstance(value, str):
                        if value.lower() in ['true', '1', 'yes']:
                            cleaned_options[key] = True
                        elif value.lower() in ['false', '0', 'no']:
                            cleaned_options[key] = False
                        else:
                            return {"success": False, "error": f"Invalid {key}: must be boolean"}
                    else:
                        return {"success": False, "error": f"Invalid {key}: must be boolean"}
                
                else:
                    # For other options, just pass through but log
                    cleaned_options[key] = value
                    logger.info(f"Passing through option {key}={value}")
        
        return {"success": True, "options": cleaned_options}
    
    def _load_csv(self, filepath: Path, **options) -> pd.DataFrame:
        """Load CSV file"""
        # Set defaults if not provided
        if 'delimiter' not in options and 'sep' not in options:
            # Try to detect delimiter
            try:
                with open(filepath, 'r', encoding=options.get('encoding', 'utf-8')) as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        options['delimiter'] = '\t'
                    elif ';' in first_line and first_line.count(';') > first_line.count(','):
                        options['delimiter'] = ';'
            except Exception as e:
                logger.warning(f"Failed to detect delimiter: {e}")
        
        logger.info(f"Loading CSV with options: {options}")
        return pd.read_csv(filepath, **options)
    
    def _load_tsv(self, filepath: Path, **options) -> pd.DataFrame:
        """Load TSV file"""
        options.setdefault('delimiter', '\t')
        logger.info(f"Loading TSV with options: {options}")
        return pd.read_csv(filepath, **options)
    
    def _load_excel(self, filepath: Path, **options) -> pd.DataFrame:
        """Load Excel file"""
        import os
        file_ext = os.path.splitext(filepath)[1].lower()
        if 'engine' not in options:
            if file_ext == '.xls':
                options['engine'] = 'xlrd'
            elif file_ext in ['.xlsx', '.xlsm', '.xlsb']:
                options['engine'] = 'openpyxl'
        
        try:
            return pd.read_excel(filepath, **options)
        
        except ImportError as e:
            if 'xlrd' in str(e):
                raise ImportError("xlrd library is required for .xls files. Install it with: pip install xlrd")
            elif 'openpyxl' in str(e):
                raise ImportError("openpyxl library is required for .xlsx files. Install it with: pip install openpyxl")
            else:
                raise
        logger.info(f"Loading Excel with options: {options}")
        return pd.read_excel(filepath, **options)
    
    def _load_json(self, filepath: Path, **options) -> pd.DataFrame:
        """Load JSON file"""
        logger.info(f"Loading JSON with options: {options}")
        
        # If no orient specified, try to detect
        if 'orient' not in options:
            try:
                # Try records first (most common)
                df = pd.read_json(filepath, orient='records', **{k: v for k, v in options.items() if k != 'orient'})
                return df
            except:
                try:
                    # Try lines format
                    options['lines'] = True
                    return pd.read_json(filepath, **options)
                except:
                    # Fall back to default
                    pass
        
        return pd.read_json(filepath, **options)
    
    def _load_parquet(self, filepath: Path, **options) -> pd.DataFrame:
        """Load Parquet file"""
        logger.info(f"Loading Parquet with options: {options}")
        return pd.read_parquet(filepath, **options)
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """
        Get information about supported file formats.
        
        Returns:
            Dictionary with supported formats and their capabilities
        """
        return {
            "supported_extensions": list(self.loaders.keys()),
            "format_info": {
                ".csv": {
                    "description": "Comma-separated values",
                    "common_options": ["delimiter", "encoding", "header", "skiprows", "nrows"],
                    "auto_detection": ["delimiter", "encoding"]
                },
                ".tsv": {
                    "description": "Tab-separated values",
                    "common_options": ["encoding", "header", "skiprows", "nrows"],
                    "auto_detection": ["encoding"]
                },
                ".xlsx": {
                    "description": "Excel workbook",
                    "common_options": ["sheet_name", "header", "skiprows", "nrows"],
                    "auto_detection": ["sheets"]
                },
                ".xls": {
                    "description": "Excel workbook (legacy)",
                    "common_options": ["sheet_name", "header", "skiprows", "nrows"],
                    "auto_detection": ["sheets"]
                },
                ".json": {
                    "description": "JSON data",
                    "common_options": ["orient", "lines"],
                    "auto_detection": ["orient"]
                },
                ".parquet": {
                    "description": "Apache Parquet columnar storage",
                    "common_options": ["columns"],
                    "auto_detection": ["schema"]
                }
            }
        }
    
    def preview_file(self, filepath: str, **options) -> Dict[str, Any]:
        """
        Preview a file without loading it fully.
        
        Args:
            filepath: Path to the file
            **options: Loading options
            
        Returns:
            Preview information
        """
        # Validate filepath
        is_valid, clean_filepath = self.input_validator.validate_filepath(filepath)
        if not is_valid:
            return {"success": False, "error": f"Invalid filepath: {clean_filepath}"}
        
        # Validate options
        validation_result = self._validate_and_clean_options(options)
        if not validation_result["success"]:
            return validation_result
        cleaned_options = validation_result["options"]
        
        filepath = Path(clean_filepath)
        file_ext = filepath.suffix.lower()
        
        if file_ext not in self.loaders:
            return {"success": False, "error": f"Unsupported file type: {file_ext}"}
        
        try:
            # Load just a small sample
            preview_options = cleaned_options.copy()
            preview_options['nrows'] = 10  # Limit to 10 rows for preview
            
            df_preview = self.loaders[file_ext](filepath, **preview_options)
            
            return {
                "success": True,
                "preview": {
                    "shape": df_preview.shape,
                    "columns": df_preview.columns.tolist(),
                    "dtypes": {col: str(dtype) for col, dtype in df_preview.dtypes.items()},
                    "sample_data": df_preview.head().to_dict('records'),
                    "file_info": {
                        "path": str(filepath),
                        "size_mb": round(filepath.stat().st_size / 1024 / 1024, 2),
                        "type": file_ext.replace('.', '')
                    }
                },
                "options_used": preview_options
            }
            
        except Exception as e:
            return {"success": False, "error": f"Failed to preview file: {str(e)}"}