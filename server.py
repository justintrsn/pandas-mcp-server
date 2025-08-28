#!/usr/bin/env python3
"""
Pandas MCP Server - THIN LAYER
This is just the MCP interface layer that calls functions from core.tools
"""

import sys
import os
import logging
import tempfile
import base64
import threading
import time
import glob
from typing import Dict, List, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP

# Import ALL orchestration functions from core.tools
from core.tools import (
    read_metadata,
    run_pandas_code,
    load_dataframe,
    list_dataframes,
    get_dataframe_info,
    validate_pandas_code,
    get_execution_context,
    preview_file,
    get_supported_formats,
    clear_session,
    get_session_info,
    get_server_info,
    create_chart,
    suggest_charts,
    get_chart_types,
    create_correlation_heatmap,
    create_time_series_chart
)

# Import configuration
from core.config import (
    SERVER_NAME,
    SERVER_VERSION,
    SERVER_DESCRIPTION,
    SERVER_HOST,
    SERVER_PORT,
    SERVER_TRANSPORT,
    LOG_LEVEL,
    LOG_FILE,
    LOG_FORMAT
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server - name only (no version parameter)
mcp = FastMCP(SERVER_NAME)

# ============================================================================
# CLIENT-SIDE FILE MANAGEMENT SUPPORT
# ============================================================================

@mcp.tool()
async def upload_temp_file_tool(
    filename: str,
    content: str,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Upload a temporary file for processing.
    Fixed to save where load_dataframe_tool can find it.
    """
    try:
        from pathlib import Path
        
        # Save to current working directory (where load_dataframe expects files)
        # Or use data directory from config
        save_path = Path(f"temp_{session_id}_{filename}")
        
        # Write the file
        save_path.write_text(content, encoding='utf-8')
        
        return {
            "success": True,
            "filepath": str(save_path),  # Return the path that load_dataframe can use
            "filename": filename,
            "session_id": session_id,
            "size_bytes": len(content.encode('utf-8')),
            "message": f"File uploaded as {save_path}. Use filepath '{save_path}' with load_dataframe_tool."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# MCP TOOL DEFINITIONS - THIN WRAPPERS (EXISTING TOOLS)
# ============================================================================

@mcp.tool()
async def read_metadata_tool(
    filepath: str,
    sample_size: Optional[int] = 1000
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from data files (CSV, Excel, JSON, Parquet).
    
    This tool provides detailed analysis including:
    - File structure and encoding information
    - Column-by-column data type analysis
    - Data quality metrics and issues
    - Statistical summaries
    - Memory usage and optimization suggestions
    - Relationship detection between columns
    - Actionable recommendations for data preparation
    
    Args:
        filepath: Path to the data file to analyze (can be temporary file from upload_temp_file_tool)
        sample_size: Number of rows to sample for large files (default: 1000)
    
    Returns:
        Comprehensive metadata dictionary
    """
    # Just call the orchestration function from tools.py
    return read_metadata(filepath, sample_size)


@mcp.tool()
async def run_pandas_code_tool(
    code: str,
    target_df: Optional[str] = "df",
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Execute pandas operations on DataFrames with security validation.
    
    This tool allows execution of pandas code with:
    - Security filtering to prevent malicious operations
    - Access to loaded DataFrames in the session
    - Support for data transformations and analysis
    - Automatic result type detection
    
    Args:
        code: Pandas code to execute
        target_df: Name of the target DataFrame (default: "df")
        session_id: Session identifier (default: "default")
    
    Returns:
        Execution result with success status
    """
    # Just call the orchestration function from tools.py
    return run_pandas_code(code, target_df, session_id)


@mcp.tool()
async def load_dataframe_tool(
    filepath: str,
    df_name: Optional[str] = "df",
    session_id: Optional[str] = "default",
    sheet_name: Optional[str] = None,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load a data file into a DataFrame for analysis.
    
    Works with both regular files and temporary files from upload_temp_file_tool.
    
    Args:
        filepath: Path to the file to load (can be temp file path from upload_temp_file_tool)
        df_name: Name to assign to the DataFrame (default: "df")
        session_id: Session identifier (default: "default")
        sheet_name: For Excel files, specify sheet name
        delimiter: For CSV files, specify delimiter
        encoding: File encoding (auto-detected if not specified)
    
    Returns:
        Success status and DataFrame information
    """
    # Just call the orchestration function from tools.py
    return load_dataframe(
        filepath=filepath,
        df_name=df_name,
        session_id=session_id,
        sheet_name=sheet_name,
        delimiter=delimiter,
        encoding=encoding
    )


@mcp.tool()
async def list_dataframes_tool(
    session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    List all loaded DataFrames with their metadata.
    
    Args:
        session_id: Optional session ID to filter by
    
    Returns:
        List of DataFrames with their properties
    """
    # Just call the orchestration function from tools.py
    return list_dataframes(session_id)


@mcp.tool()
async def get_dataframe_info_tool(
    df_name: str,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Get detailed information about a specific DataFrame.
    
    Args:
        df_name: Name of the DataFrame
        session_id: Session identifier (default: "default")
    
    Returns:
        DataFrame information including shape, columns, dtypes, memory usage
    """
    # Just call the orchestration function from tools.py
    return get_dataframe_info(df_name, session_id)


@mcp.tool()
async def validate_pandas_code_tool(
    code: str,
    target_df: Optional[str] = "df",
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Validate pandas code without executing it.
    
    Args:
        code: Pandas code to validate
        target_df: Target DataFrame name (default: "df")
        session_id: Session identifier (default: "default")
    
    Returns:
        Validation result with success status
    """
    # Just call the orchestration function from tools.py
    return validate_pandas_code(code, target_df, session_id)


@mcp.tool()
async def get_execution_context_tool(
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Get information about the execution context.
    
    Args:
        session_id: Session identifier (default: "default")
    
    Returns:
        Context information including available DataFrames and functions
    """
    # Just call the orchestration function from tools.py
    return get_execution_context(session_id)


@mcp.tool()
async def preview_file_tool(
    filepath: str,
    delimiter: Optional[str] = None,
    encoding: Optional[str] = None
) -> Dict[str, Any]:
    """
    Preview a file without fully loading it.
    
    Args:
        filepath: Path to the file (can be temp file from upload_temp_file_tool)
        delimiter: CSV delimiter (optional)
        encoding: File encoding (optional)
    
    Returns:
        Preview information including sample data and detected parameters
    """
    # Build kwargs from optional parameters
    kwargs = {}
    if delimiter is not None:
        kwargs['delimiter'] = delimiter
    if encoding is not None:
        kwargs['encoding'] = encoding
    
    # Just call the orchestration function from tools.py
    return preview_file(filepath, **kwargs)


@mcp.tool()
async def get_supported_formats_tool() -> Dict[str, Any]:
    """
    Get information about supported file formats.
    
    Returns:
        Dictionary with supported formats and their capabilities
    """
    # Just call the orchestration function from tools.py
    return get_supported_formats()


@mcp.tool()
async def clear_session_tool(
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Clear all data in a session including temporary files.
    
    Args:
        session_id: Session identifier (default: "default")
    
    Returns:
        Clear operation result
    """
    # Just call the orchestration function from tools.py
    return clear_session(session_id)


@mcp.tool()
async def get_session_info_tool(
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Get information about a session.
    
    Args:
        session_id: Session identifier (default: "default")
    
    Returns:
        Session information including DataFrames and memory usage
    """
    # Just call the orchestration function from tools.py
    return get_session_info(session_id)


@mcp.tool()
async def get_server_info_tool() -> Dict[str, Any]:
    """
    Get information about the MCP server configuration and status.
    
    Returns:
        Server configuration and status information
    """
    # Just call the orchestration function from tools.py
    return get_server_info()

# ============================================================================
# Visualization Tools
# ============================================================================

@mcp.tool()
async def create_chart_tool(
    df_name: str,
    chart_type: str,
    x_column: Optional[str] = None,
    y_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: Optional[str] = "default",
    width: Optional[int] = None,
    height: Optional[int] = None,
    group_by: Optional[str] = None,
    aggregate: Optional[str] = None,
    max_points: Optional[int] = None,
    show_trend: Optional[bool] = None,
    color_scheme: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create an interactive HTML chart from a DataFrame.
    
    Generates beautiful, interactive charts with customization controls.
    Supported chart types:
    - bar: Compare values across categories (grouped, stacked, horizontal)
    - line: Show trends over time or continuous data
    - pie: Display proportions (also supports doughnut style)
    - scatter: Reveal relationships between variables (supports bubble charts)
    - heatmap: Visualize matrix data or correlations
    
    Args:
        df_name: Name of the DataFrame to visualize
        chart_type: Type of chart (bar, line, pie, scatter, heatmap)
        x_column: Column for x-axis/labels (auto-detected if not specified)
        y_columns: List of columns for y-axis/values (auto-detected if not specified)
        title: Chart title
        session_id: Session identifier (default: "default")
        width: Chart width in pixels
        height: Chart height in pixels
        group_by: Column to group by for aggregation (bar charts)
        aggregate: Aggregation function (mean, sum, count, etc.)
        max_points: Maximum number of data points to display
        show_trend: Show trend lines (scatter charts)
        color_scheme: Color scheme (viridis, coolwarm, grayscale for heatmaps)
    
    Returns:
        Dictionary with chart file path and visualization metadata
    """
    # Build options dictionary from optional parameters
    options = {}
    if width is not None:
        options['width'] = width
    if height is not None:
        options['height'] = height
    if group_by is not None:
        options['group_by'] = group_by
    if aggregate is not None:
        options['aggregate'] = aggregate
    if max_points is not None:
        options['max_points'] = max_points
    if show_trend is not None:
        options['show_trend'] = show_trend
    if color_scheme is not None:
        options['color_scheme'] = color_scheme
    
    # Call the synchronous function from tools.py
    return create_chart(
        df_name=df_name,
        chart_type=chart_type,
        x_column=x_column,
        y_columns=y_columns,
        title=title,
        session_id=session_id,
        **options
    )


@mcp.tool()
async def suggest_charts_tool(
    df_name: str,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Get intelligent chart suggestions based on DataFrame characteristics.
    
    Analyzes the DataFrame's structure, column types, and data distribution
    to recommend the most appropriate visualizations with optimal configurations.
    
    Args:
        df_name: Name of the DataFrame to analyze
        session_id: Session identifier (default: "default")
    
    Returns:
        Dictionary with ranked chart suggestions and recommended configurations
    """
    return suggest_charts(df_name, session_id)


@mcp.tool()
async def get_chart_types_tool() -> Dict[str, Any]:
    """
    Get detailed information about all supported chart types.
    
    Returns comprehensive documentation for each chart type including:
    - Description and best use cases
    - Required and optional data formats
    - Available customization options
    - Example configurations
    
    Returns:
        Dictionary with all supported chart types and their specifications
    """
    return get_chart_types()


@mcp.tool()
async def create_correlation_heatmap_tool(
    df_name: str,
    columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: Optional[str] = "default",
    color_scheme: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a correlation heatmap for analyzing relationships between numeric columns.
    
    Generates an interactive heatmap showing correlation coefficients between
    all numeric columns or a specified subset. Perfect for feature analysis
    and identifying multicollinearity.
    
    Args:
        df_name: Name of the DataFrame
        columns: Specific columns to include (uses all numeric if not specified)
        title: Chart title
        session_id: Session identifier (default: "default")
        color_scheme: Color scheme (viridis, coolwarm, grayscale)
    
    Returns:
        Dictionary with heatmap file path and correlation statistics
    """
    options = {}
    if color_scheme:
        options['color_scheme'] = color_scheme
    
    return create_correlation_heatmap(
        df_name=df_name,
        columns=columns,
        title=title,
        session_id=session_id,
        **options
    )


@mcp.tool()
async def create_time_series_chart_tool(
    df_name: str,
    time_column: Optional[str] = None,
    value_columns: Optional[List[str]] = None,
    title: Optional[str] = None,
    session_id: Optional[str] = "default",
    fill: Optional[bool] = None,
    tension: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create an optimized time series line chart.
    
    Automatically detects datetime columns and creates a properly formatted
    time series visualization with intelligent resampling for large datasets.
    
    Args:
        df_name: Name of the DataFrame
        time_column: Time/date column (auto-detected if not specified)
        value_columns: Value columns to plot (auto-detected if not specified)
        title: Chart title
        session_id: Session identifier (default: "default")
        fill: Fill area under lines (default: False)
        tension: Line curve tension 0-0.5 (default: 0.2)
    
    Returns:
        Dictionary with chart file path and time series metadata
    """
    options = {}
    if fill is not None:
        options['fill'] = fill
    if tension is not None:
        options['tension'] = tension
    
    return create_time_series_chart(
        df_name=df_name,
        time_column=time_column,
        value_columns=value_columns,
        title=title,
        session_id=session_id,
        **options
    )


# ============================================================================
# BACKGROUND CLEANUP TASK
# ============================================================================

def start_cleanup_task():
    """Start background task to clean up expired sessions and orphaned temp files"""
    def cleanup_loop():
        while True:
            try:
                # Clean up expired sessions (which also cleans their temp files)
                from storage.dataframe_manager import get_manager
                manager = get_manager()
                manager._cleanup_expired_sessions()
                
                # Clean up orphaned temp files older than 1 hour
                temp_pattern = os.path.join(tempfile.gettempdir(), "mcp_temp_*")
                current_time = time.time()
                
                for temp_file in glob.glob(temp_pattern):
                    try:
                        if os.path.exists(temp_file):
                            age = current_time - os.path.getctime(temp_file)
                            if age > 3600:  # 1 hour
                                os.unlink(temp_file)
                                logger.info(f"Cleaned up orphaned temp file: {temp_file}")
                    except Exception as e:
                        logger.debug(f"Could not clean temp file {temp_file}: {e}")
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
                time.sleep(60)  # Wait before retrying
    
    # Start cleanup in background thread
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Background cleanup task started")


# ============================================================================
# MAIN SERVER FUNCTION
# ============================================================================

def main():
    """Main entry point - starts the MCP server"""
    try:
        # Print startup banner
        print("=" * 60)
        print(f" {SERVER_NAME} v{SERVER_VERSION}")
        print(f" {SERVER_DESCRIPTION}")
        print("=" * 60)
        print()
        
        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
        logger.info(f"Server configuration: {SERVER_TRANSPORT} on {SERVER_HOST}:{SERVER_PORT}")
        
        # Start background cleanup task
        start_cleanup_task()
        
        # Log available tools (including new upload tool)
        logger.info("Available MCP tools:")
        logger.info("  - upload_temp_file_tool: Upload files temporarily for processing")
        logger.info("  - read_metadata_tool: Extract comprehensive file metadata")
        logger.info("  - run_pandas_code_tool: Execute pandas operations")
        logger.info("  - load_dataframe_tool: Load data files")
        logger.info("  - list_dataframes_tool: List loaded DataFrames")
        logger.info("  - get_dataframe_info_tool: Get DataFrame details")
        logger.info("  - validate_pandas_code_tool: Validate code without execution")
        logger.info("  - get_execution_context_tool: Get execution context")
        logger.info("  - preview_file_tool: Preview files")
        logger.info("  - get_supported_formats_tool: Get supported formats")
        logger.info("  - clear_session_tool: Clear session data")
        logger.info("  - get_session_info_tool: Get session info")
        logger.info("  - get_server_info_tool: Get server configuration")
        logger.info("  - create_chart_tool: Create interactive charts")
        logger.info("  - suggest_charts_tool: Get chart recommendations")
        logger.info("  - get_chart_types_tool: List supported chart types")
        logger.info("  - create_correlation_heatmap_tool: Create correlation matrix")
        logger.info("  - create_time_series_chart_tool: Create time series visualization")
                
        # Start server based on transport
        if SERVER_TRANSPORT == "sse":
            # SSE transport for HTTP
            print(f"Server running at http://{SERVER_HOST}:{SERVER_PORT}")
            print("Connect your MCP client to this URL")
            print()
            print()
            print("Available tools:")
            print("  - upload_temp_file_tool")
            print("  - read_metadata_tool")
            print("  - run_pandas_code_tool")
            print("  - load_dataframe_tool")
            print("  - list_dataframes_tool")
            print("  - get_dataframe_info_tool")
            print("  - validate_pandas_code_tool")
            print("  - get_execution_context_tool")
            print("  - preview_file_tool")
            print("  - get_supported_formats_tool")
            print("  - clear_session_tool")
            print("  - get_session_info_tool")
            print("  - get_server_info_tool")
            print("  - create_chart_tool")
            print("  - suggest_charts_tool")
            print("  - get_chart_types_tool")
            print("  - create_correlation_heatmap_tool")
            print("  - create_time_series_chart_tool")
            print()
            print("Press Ctrl+C to stop the server")
            
            # Run with SSE transport
            mcp.run(
                transport="sse",
                host=SERVER_HOST,
                port=SERVER_PORT
            )
        else:
            # Default to stdio transport
            print("Running in stdio mode")
            print("Connect via stdio transport")
            print()
            mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        print("\nShutting down server...")
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()