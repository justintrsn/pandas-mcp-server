"""
Core tool implementations for Pandas MCP Server.
This module contains SIMPLE orchestration logic - validation happens in the core modules.
"""

import logging
from typing import Dict, Any, Optional

# Import from core modules - each handles its own validation
from core.metadata import MetadataExtractor
from core.execution import PandasExecutor
from core.data_loader import DataLoader
from storage.dataframe_manager import get_manager
from core.config import get_config

logger = logging.getLogger(__name__)

# Initialize shared components - these handle their own validation
metadata_extractor = MetadataExtractor()
pandas_executor = PandasExecutor()
data_loader = DataLoader()
df_manager = get_manager()


def read_metadata(filepath: str, sample_size: int = 1000) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from a data file.
    
    SIMPLE ORCHESTRATION - validation happens INSIDE MetadataExtractor
    
    Args:
        filepath: Path to the data file
        sample_size: Number of rows to sample for analysis
        
    Returns:
        Dictionary containing metadata, structure, quality metrics, etc.
    """
    try:
        logger.info(f"Orchestrating metadata extraction for: {filepath}")
        
        # MetadataExtractor handles ALL validation internally
        metadata = metadata_extractor.extract_metadata(filepath, sample_size)
        
        # Log success
        file_info = metadata.get("file_info", {})
        structure = metadata.get("structure", {})
        logger.info(
            f"Successfully extracted metadata from {file_info.get('name', filepath)}: "
            f"{structure.get('rows', 0)} rows, "
            f"{structure.get('columns', 0)} columns"
        )
        
        return metadata
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return {"error": f"File not found: {str(e)}"}
    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return {"error": str(e)}
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Error extracting metadata: {e}", exc_info=True)
        return {"error": f"Failed to extract metadata: {str(e)}"}


def run_pandas_code(
    code: str,
    target_df: str = "df",
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Execute pandas code on a DataFrame with security validation.
    
    SIMPLE ORCHESTRATION - validation happens INSIDE PandasExecutor
    
    Args:
        code: Pandas code to execute
        target_df: Name of the target DataFrame
        session_id: Session identifier
        
    Returns:
        Execution result with success status
    """
    try:
        logger.info(f"Orchestrating pandas code execution in session {session_id}")
        
        # PandasExecutor handles ALL validation internally
        result = pandas_executor.execute(code, target_df, session_id)
        
        if result.get("success"):
            logger.info(f"Code executed successfully, result type: {result.get('result_type')}")
        else:
            logger.warning(f"Code execution failed: {result.get('error')}")
            
        return result
        
    except Exception as e:
        logger.error(f"Execution orchestration failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Execution failed: {str(e)}"
        }


def load_dataframe(
    filepath: str,
    df_name: str = "df",
    session_id: str = "default",
    sheet_name: Optional[str] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None,
    **additional_options
) -> Dict[str, Any]:
    """
    Load a data file into a DataFrame.
    
    SIMPLE ORCHESTRATION - validation happens INSIDE DataLoader
    
    Args:
        filepath: Path to the file
        df_name: Name to assign to the DataFrame
        session_id: Session identifier
        sheet_name: Excel sheet name (optional)
        delimiter: CSV delimiter (optional)
        encoding: File encoding (optional)
        **additional_options: Other loading options
        
    Returns:
        Success status and DataFrame information
    """
    try:
        logger.info(f"Orchestrating data loading: {filepath} -> DataFrame '{df_name}'")
        
        # Prepare options
        options = {}
        if sheet_name is not None:
            options['sheet_name'] = sheet_name
        if delimiter is not None:
            options['delimiter'] = delimiter
        if encoding is not None:
            options['encoding'] = encoding
        
        # Add any additional options
        options.update(additional_options)
        
        # DataLoader handles ALL validation internally
        result = data_loader.load(filepath, df_name, session_id, **options)
        
        if result.get("success"):
            logger.info(f"Successfully loaded {filepath} as '{df_name}'")
        else:
            logger.error(f"Failed to load {filepath}: {result.get('error')}")
            
        return result
        
    except Exception as e:
        logger.error(f"Data loading orchestration failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def list_dataframes(session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    List all loaded DataFrames.
    
    Args:
        session_id: Optional session filter
        
    Returns:
        List of DataFrames with metadata
    """
    try:
        logger.info(f"Listing DataFrames for session: {session_id or 'all'}")
        
        dataframes = df_manager.list_dataframes(session_id)
        
        return {
            "success": True,
            "dataframes": dataframes,
            "total_count": len(dataframes),
            "session_filter": session_id
        }
        
    except Exception as e:
        logger.error(f"Failed to list dataframes: {e}")
        return {"success": False, "error": str(e)}


def get_dataframe_info(df_name: str, session_id: str = "default") -> Dict[str, Any]:
    """
    Get detailed information about a specific DataFrame.
    
    Args:
        df_name: Name of the DataFrame
        session_id: Session identifier
        
    Returns:
        DataFrame information
    """
    try:
        logger.info(f"Getting info for DataFrame '{df_name}' in session '{session_id}'")
        
        df = df_manager.get_dataframe(df_name, session_id)
        if df is None:
            return {
                "success": False,
                "error": f"DataFrame '{df_name}' not found in session '{session_id}'"
            }
        
        return {
            "success": True,
            "dataframe_info": {
                "name": df_name,
                "session_id": session_id,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
                "null_counts": df.isnull().sum().to_dict(),
                "preview": df.head().to_string(max_cols=10)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get DataFrame info: {e}")
        return {"success": False, "error": str(e)}


def validate_pandas_code(
    code: str,
    target_df: str = "df",
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Validate pandas code without executing it.
    
    Args:
        code: Pandas code to validate
        target_df: Target DataFrame name
        session_id: Session identifier
        
    Returns:
        Validation result
    """
    try:
        logger.info(f"Validating pandas code for DataFrame '{target_df}'")
        
        # PandasExecutor handles validation internally
        result = pandas_executor.validate_operation_only(code, target_df, session_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Code validation failed: {e}")
        return {"valid": False, "error": str(e)}


def get_execution_context(session_id: str = "default") -> Dict[str, Any]:
    """
    Get execution context information.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Context information
    """
    try:
        logger.info(f"Getting execution context for session '{session_id}'")
        
        # PandasExecutor handles validation internally
        result = pandas_executor.get_execution_context_info(session_id)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get execution context: {e}")
        return {"error": str(e)}


def preview_file(filepath: str, **options) -> Dict[str, Any]:
    """
    Preview a file without fully loading it.
    
    Args:
        filepath: Path to the file
        **options: Loading options
        
    Returns:
        Preview information
    """
    try:
        logger.info(f"Previewing file: {filepath}")
        
        # DataLoader handles validation internally
        result = data_loader.preview_file(filepath, **options)
        
        return result
        
    except Exception as e:
        logger.error(f"File preview failed: {e}")
        return {"success": False, "error": str(e)}


def get_supported_formats() -> Dict[str, Any]:
    """
    Get information about supported file formats.
    
    Returns:
        Supported formats information
    """
    try:
        logger.info("Getting supported file formats")
        
        # DataLoader provides this info
        formats = data_loader.get_supported_formats()
        
        return {
            "success": True,
            "formats": formats
        }
        
    except Exception as e:
        logger.error(f"Failed to get supported formats: {e}")
        return {"success": False, "error": str(e)}


def clear_session(session_id: str = "default") -> Dict[str, Any]:
    """
    Clear all data in a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Clear operation result
    """
    try:
        logger.info(f"Clearing session: {session_id}")
        
        success = df_manager.clear_session(session_id)
        
        if success:
            return {
                "success": True,
                "message": f"Session '{session_id}' cleared successfully"
            }
        else:
            return {
                "success": False,
                "error": f"Session '{session_id}' not found"
            }
            
    except Exception as e:
        logger.error(f"Failed to clear session: {e}")
        return {"success": False, "error": str(e)}


def get_session_info(session_id: str = "default") -> Dict[str, Any]:
    """
    Get information about a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Session information
    """
    try:
        logger.info(f"Getting session info: {session_id}")
        
        session_info = df_manager.get_session_info(session_id)
        
        if "error" in session_info:
            return {"success": False, "error": session_info["error"]}
        
        return {
            "success": True,
            "session_info": session_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get session info: {e}")
        return {"success": False, "error": str(e)}


def get_server_info() -> Dict[str, Any]:
    """
    Get server configuration and status information.
    
    Returns:
        Server configuration and runtime status
    """
    try:
        logger.info("Getting server information")
        
        config = get_config()
        
        # Add runtime information
        config["status"] = {
            "running": True,
            "sessions": len(df_manager.sessions),
            "total_dataframes": sum(
                len(session.dataframes) 
                for session in df_manager.sessions.values()
            ),
            "memory_usage_mb": df_manager._calculate_total_memory()
        }
        
        return config
        
    except Exception as e:
        logger.error(f"Failed to get server info: {e}")
        return {"error": str(e)}