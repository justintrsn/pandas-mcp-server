"""
Core tool implementations for Pandas MCP Server.
This module contains SIMPLE orchestration logic - validation happens in the core modules.
"""

import logging
import hashlib
import shutil
import threading
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime, timedelta

# Import from core modules - each handles its own validation
from core.metadata import MetadataExtractor
from core.execution import PandasExecutor
from core.data_loader import DataLoader
from storage.dataframe_manager import get_manager
from core.config import get_config
from core.visualization import get_orchestrator

logger = logging.getLogger(__name__)

# Initialize shared components - these handle their own validation
metadata_extractor = MetadataExtractor()
pandas_executor = PandasExecutor()
data_loader = DataLoader()
df_manager = get_manager()
visualization_orchestrator = get_orchestrator()

# ============================================================================
# SESSION TRACKING FOR AUTO-CLEANUP
# ============================================================================

class SessionTracker:
    """Track session activity for time-based cleanup"""
    
    def __init__(self):
        self.sessions = {}  # {session_id: last_activity}
        self.lock = threading.Lock()
    
    def touch(self, session_id: str):
        """Update session last activity"""
        with self.lock:
            self.sessions[session_id] = datetime.now()
    
    def get_inactive_sessions(self, max_age_minutes: int = 30):
        """Get sessions inactive for more than max_age_minutes"""
        with self.lock:
            now = datetime.now()
            inactive = []
            for sid, last_activity in self.sessions.items():
                if (now - last_activity).total_seconds() > max_age_minutes * 60:
                    inactive.append(sid)
            return inactive
    
    def remove(self, session_id: str):
        """Remove session from tracking"""
        with self.lock:
            self.sessions.pop(session_id, None)

# Global session tracker
session_tracker = SessionTracker()

# ============================================================================
# FILE UPLOAD WITH AUTO-CLEANUP
# ============================================================================

def upload_temp_file(
    filename: str,
    content: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Upload with deduplication and session tracking.
    Files auto-cleaned after 30 min of inactivity.
    """
    try:
        # Track session activity
        session_tracker.touch(session_id)
        
        # Create session directory
        session_dir = Path(f"sessions/{session_id}")
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session metadata file
        metadata_file = session_dir / ".metadata.json"
        metadata = {}
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text())
        
        metadata["last_activity"] = datetime.now().isoformat()
        metadata["session_id"] = session_id
        
        # Check for duplicate file
        filepath = session_dir / filename
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        if filepath.exists():
            existing_content = filepath.read_text(encoding='utf-8')
            existing_hash = hashlib.md5(existing_content.encode()).hexdigest()
            
            if existing_hash == content_hash:
                logger.info(f"File {filename} already exists with same content")
                metadata["files"] = metadata.get("files", {})
                metadata["files"][filename] = {"hash": content_hash, "cached_hit": True}
                
                metadata_file.write_text(json.dumps(metadata, indent=2))
                
                return {
                    "success": True,
                    "filepath": str(filepath),
                    "cached": True,
                    "message": f"Using cached file at '{filepath}'"
                }
        
        # Write new file
        filepath.write_text(content, encoding='utf-8')
        
        # Update metadata
        metadata["files"] = metadata.get("files", {})
        metadata["files"][filename] = {
            "hash": content_hash,
            "uploaded_at": datetime.now().isoformat()
        }
        
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        logger.info(f"Uploaded {filename} to session {session_id}")
        
        return {
            "success": True,
            "filepath": str(filepath),
            "cached": False,
            "message": f"File uploaded to '{filepath}'"
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"success": False, "error": str(e)}

def cleanup_session_files(session_id: str = "default") -> Dict[str, Any]:
    """
    Clean up session files and remove from tracking.
    Called by the auto-cleanup task.
    """
    try:
        session_dir = Path(f"sessions/{session_id}")
        
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Cleaned session directory: {session_dir}")
        
        # Remove from tracker
        session_tracker.remove(session_id)
        
        return {
            "success": True,
            "message": f"Session '{session_id}' cleaned"
        }
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"success": False, "error": str(e)}
    

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

def create_chart(
    df_name: str,
    chart_type: str,
    x_column: Optional[str] = None,
    y_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: str = "default",
    **options
) -> Dict[str, Any]:
    """
    Create an interactive HTML chart from a DataFrame.
    
    SIMPLE ORCHESTRATION - validation happens INSIDE VisualizationOrchestrator
    
    Args:
        df_name: Name of the DataFrame to visualize
        chart_type: Type of chart (bar, line, pie, scatter, heatmap)
        x_column: Column for x-axis/labels (optional)
        y_columns: Column(s) for y-axis/values (optional)
        title: Chart title (optional)
        session_id: Session identifier
        **options: Additional chart-specific options
        
    Returns:
        Dictionary with chart file path and metadata
    """
    try:
        logger.info(f"Orchestrating chart creation: {chart_type} for DataFrame '{df_name}'")
        
        # VisualizationOrchestrator handles ALL validation internally
        result = visualization_orchestrator.create_visualization(
            df_name=df_name,
            chart_type=chart_type,
            x_column=x_column,
            y_columns=y_columns,
            session_id=session_id,
            title=title,
            **options
        )
        
        if result.get("success"):
            logger.info(f"Chart created successfully: {result.get('filepath')}")
        else:
            logger.error(f"Chart creation failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Chart creation orchestration failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def suggest_charts(
    df_name: str,
    session_id: str = "default"
) -> Dict[str, Any]:
    """
    Get chart type suggestions based on DataFrame characteristics.
    
    SIMPLE ORCHESTRATION - validation happens INSIDE VisualizationOrchestrator
    
    Args:
        df_name: Name of the DataFrame to analyze
        session_id: Session identifier
        
    Returns:
        Dictionary with chart suggestions and configurations
    """
    try:
        logger.info(f"Orchestrating chart suggestions for DataFrame '{df_name}'")
        
        # VisualizationOrchestrator handles ALL validation internally
        result = visualization_orchestrator.suggest_charts(df_name, session_id)
        
        if result.get("success"):
            suggestions_count = len(result.get("suggestions", []))
            logger.info(f"Generated {suggestions_count} chart suggestions")
        else:
            logger.error(f"Failed to generate suggestions: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Chart suggestion orchestration failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def get_chart_types() -> Dict[str, Any]:
    """
    Get information about all supported chart types.
    
    Returns:
        Dictionary with supported chart types and their details
    """
    try:
        logger.info("Getting supported chart types")
        
        # VisualizationOrchestrator provides this info
        chart_info = visualization_orchestrator.get_supported_charts()
        
        return {
            "success": True,
            **chart_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get chart types: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def create_correlation_heatmap(
    df_name: str,
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: str = "default",
    **options
) -> Dict[str, Any]:
    """
    Create a correlation heatmap for numeric columns.
    
    Convenience function for creating correlation matrices.
    
    Args:
        df_name: Name of the DataFrame
        columns: Specific columns to include (optional, uses all numeric if not specified)
        title: Chart title (optional)
        session_id: Session identifier
        **options: Additional options
        
    Returns:
        Dictionary with chart file path and metadata
    """
    try:
        logger.info(f"Orchestrating correlation heatmap for DataFrame '{df_name}'")
        
        # Set heatmap-specific options
        options['heatmap_type'] = 'correlation'
        
        result = visualization_orchestrator.create_visualization(
            df_name=df_name,
            chart_type='heatmap',
            x_column=None,
            y_columns=columns,
            session_id=session_id,
            title=title or f"Correlation Matrix - {df_name}",
            **options
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Correlation heatmap orchestration failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


def create_time_series_chart(
    df_name: str,
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: str = "default",
    **options
) -> Dict[str, Any]:
    """
    Create a time series line chart.
    
    Convenience function for time series visualization.
    
    Args:
        df_name: Name of the DataFrame
        time_column: Time/date column (optional, auto-detected)
        value_columns: Value columns to plot (optional, auto-detected)
        title: Chart title (optional)
        session_id: Session identifier
        **options: Additional options
        
    Returns:
        Dictionary with chart file path and metadata
    """
    try:
        logger.info(f"Orchestrating time series chart for DataFrame '{df_name}'")
        
        # Set line chart options for time series
        options['fill'] = options.get('fill', False)
        options['tension'] = options.get('tension', 0.2)
        
        result = visualization_orchestrator.create_visualization(
            df_name=df_name,
            chart_type='line',
            x_column=time_column,
            y_columns=value_columns,
            session_id=session_id,
            title=title or f"Time Series - {df_name}",
            **options
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Time series chart orchestration failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}