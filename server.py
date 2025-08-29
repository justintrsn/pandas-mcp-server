#!/usr/bin/env python3
"""
Pandas MCP Server - THIN LAYER
This is just the MCP interface layer that calls functions from core.tools
"""

"""
Pandas MCP Server - With Health Endpoints and Auto-Cleanup
"""

import sys
import os
import logging
import tempfile
import threading
import time
import glob
import shutil
import json
from typing import Dict, Optional, Any, List
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP
from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse

# Import ALL orchestration functions AND session_tracker from core.tools
from core.tools import (
    upload_temp_file,
    session_tracker,  
    read_metadata,
    run_pandas_code,
    load_dataframe,
    list_dataframes,
    store_dataframe,
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
    create_time_series_chart,
    get_chart_html
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

# Initialize FastMCP server
mcp = FastMCP(SERVER_NAME)

# Create FastAPI app for custom endpoints
app = FastAPI(title=SERVER_NAME, version=SERVER_VERSION)

# Track server start time for status
server_start_time = time.time()

# ============================================================================
# SERVER STATUS
# ============================================================================
def detailed_status():
    """Detailed status endpoint with session information"""
    from storage.dataframe_manager import get_manager
    manager = get_manager()
    
    # Get session information
    sessions_dir = Path("sessions")
    active_sessions = []
    total_files = 0
    total_size_mb = 0
    
    if sessions_dir.exists():
        for session_dir in sessions_dir.iterdir():
            if session_dir.is_dir():
                files = list(session_dir.glob("*"))
                file_count = len(files)
                total_files += file_count
                
                # Calculate size
                session_size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
                total_size_mb += session_size
                
                # Get metadata
                metadata_file = session_dir / ".metadata.json"
                last_activity = None
                if metadata_file.exists():
                    try:
                        metadata = json.loads(metadata_file.read_text())
                        last_activity = metadata.get("last_activity")
                    except:
                        pass
                
                # Calculate age
                age_minutes = None
                if last_activity:
                    try:
                        last_time = datetime.fromisoformat(last_activity)
                        age_minutes = int((datetime.now() - last_time).total_seconds() / 60)
                    except:
                        pass
                
                active_sessions.append({
                    "session_id": session_dir.name,
                    "files": file_count,
                    "size_mb": round(session_size, 2),
                    "last_activity": last_activity,
                    "age_minutes": age_minutes,
                    "will_expire_in": max(0, 30 - (age_minutes or 0)) if age_minutes is not None else None
                })
    
    # Sort by last activity
    active_sessions.sort(key=lambda x: x.get("last_activity") or "", reverse=True)
    
    # Get tracked sessions from session_tracker
    tracked_count = len(session_tracker.sessions) if hasattr(session_tracker, 'sessions') else 0
    
    uptime_seconds = int(time.time() - server_start_time)
    
    return JSONResponse({
        "server": {
            "name": SERVER_NAME,
            "version": SERVER_VERSION,
            "description": SERVER_DESCRIPTION,
            "uptime": {
                "seconds": uptime_seconds,
                "human_readable": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m"
            },
            "started_at": datetime.fromtimestamp(server_start_time).isoformat()
        },
        "configuration": {
            "host": SERVER_HOST,
            "port": SERVER_PORT,
            "transport": SERVER_TRANSPORT,
            "auto_cleanup": {
                "enabled": True,
                "timeout_minutes": 30,
                "check_interval_minutes": 5
            }
        },
        "sessions": {
            "active_count": len(active_sessions),
            "tracked_count": tracked_count,
            "total_files": total_files,
            "total_size_mb": round(total_size_mb, 2),
            "details": active_sessions
        },
        "dataframes": {
            "loaded_count": sum(
                len(session.dataframes) 
                for session in manager.sessions.values()
            ),
            "memory_usage_mb": manager._calculate_total_memory() if hasattr(manager, '_calculate_total_memory') else 0
        },
        "timestamp": datetime.now().isoformat()
    })

# ============================================================================
# AUTO-CLEANUP BACKGROUND TASK
# ============================================================================

def start_auto_cleanup_task():
    """Background task that automatically cleans inactive sessions"""
    def cleanup_loop():
        while True:
            try:
                # Get sessions inactive for 30+ minutes
                inactive_sessions = session_tracker.get_inactive_sessions(30)
                
                for session_id in inactive_sessions:
                    # Clean session directory
                    session_dir = Path(f"sessions/{session_id}")
                    if session_dir.exists():
                        shutil.rmtree(session_dir)
                        logger.info(f"Auto-cleaned inactive session: {session_id} (30 min timeout)")
                    
                    # Clear from DataFrame manager
                    from storage.dataframe_manager import get_manager
                    manager = get_manager()
                    manager.clear_session(session_id)
                    
                    # Remove from tracker
                    session_tracker.remove(session_id)
                
                if inactive_sessions:
                    logger.info(f"Cleaned {len(inactive_sessions)} inactive sessions")
                
                # Clean orphaned session directories
                sessions_dir = Path("sessions")
                if sessions_dir.exists():
                    for session_dir in sessions_dir.iterdir():
                        if session_dir.is_dir():
                            age_hours = (time.time() - session_dir.stat().st_mtime) / 3600
                            if age_hours > 1:  # Older than 1 hour
                                session_id = session_dir.name
                                if session_id not in session_tracker.sessions:
                                    shutil.rmtree(session_dir)
                                    logger.info(f"Cleaned orphaned session directory: {session_id}")
                
                # Clean old temp files
                temp_pattern = os.path.join(tempfile.gettempdir(), "mcp_temp_*")
                for temp_file in glob.glob(temp_pattern):
                    if os.path.exists(temp_file):
                        age = time.time() - os.path.getctime(temp_file)
                        if age > 3600:  # 1 hour
                            os.unlink(temp_file)
                            logger.debug(f"Cleaned old temp file: {temp_file}")
                
                time.sleep(300)  # Run every 5 minutes
                
            except Exception as e:
                logger.error(f"Auto-cleanup error: {e}")
                time.sleep(60)

    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info("Auto-cleanup task started (30 min session timeout)")

# ============================================================================
# MCP TOOL DEFINITIONS - THIN WRAPPERS 
# ============================================================================
@mcp.tool()
async def upload_temp_file_tool(
    filename: str,
    content: str,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Upload a file for data analysis with intelligent deduplication.
    
    Files are stored in session-specific directories (sessions/{session_id}/).
    If the exact same file already exists, returns the existing path without re-uploading.
    
    Args:
        filename: Name of the file (e.g., 'sales_data.csv')
        content: File content as string
        session_id: Session identifier for isolation (default: 'default')
    
    Returns:
        Dict with filepath to use with load_dataframe_tool and cached status
    """
    return upload_temp_file(filename, content, session_id)

@mcp.tool()
async def get_server_status_tool() -> Dict[str, Any]:
    """Get server status via MCP tool"""
    response = await detailed_status()
    return json.loads(response.body)

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
async def store_dataframe_tool(
    source_df_name: str,
    new_df_name: str,
    session_id: Optional[str] = "default"
) -> Dict[str, Any]:
    """
    Explicitly store a DataFrame result in the session.
    
    Args:
        source_df_name: Name of the source DataFrame variable to store
        new_df_name: New name for the stored DataFrame
        session_id: Session identifier
    
    Returns:
        Success status and message
    """
    return store_dataframe(source_df_name, new_df_name, session_id)

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

@mcp.tool()
async def get_chart_html_tool(
    filepath: str
) -> Dict[str, Any]:
    """
    Retrieve HTML content of a generated chart file.
    
    Args:
        filepath: Path to the chart HTML file (from create_chart_tool response)
    
    Returns:
        Dictionary with HTML content and metadata
    """
    return get_chart_html(filepath)

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
    """Main entry point - starts the MCP server with health endpoints"""
    try:
        # Print startup banner
        print("=" * 60)
        print(f" {SERVER_NAME} v{SERVER_VERSION}")
        print(f" {SERVER_DESCRIPTION}")
        print("=" * 60)
        print()
        print("AUTO-CLEANUP: Sessions expire after 30 minutes of inactivity")
        print("=" * 60)
        print()
        
        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
        logger.info(f"Server configuration: {SERVER_TRANSPORT} on {SERVER_HOST}:{SERVER_PORT}")
        
        # Start BOTH cleanup tasks
        start_cleanup_task()  # Original cleanup for DataFrame manager
        start_auto_cleanup_task()  # New auto-cleanup for sessions
        
        # Log available tools
        logger.info("Session management: Automatic cleanup enabled (30 min timeout)")
        logger.info("Available MCP tools:")
        logger.info("  - upload_temp_file_tool: Upload files with auto-cleanup")
        logger.info("  - get_server_status_tool: Get server status")
        logger.info("  - read_metadata_tool: Extract file metadata")
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
            print(f"🚀 Server running at http://{SERVER_HOST}:{SERVER_PORT}")
            print()
            print("📍 HTTP Endpoints:")
            print(f"   • Base:   http://{SERVER_HOST}:{SERVER_PORT}/")
            print(f"   • Health: http://{SERVER_HOST}:{SERVER_PORT}/health")
            print(f"   • Status: http://{SERVER_HOST}:{SERVER_PORT}/status")
            print(f"   • MCP:    http://{SERVER_HOST}:{SERVER_PORT}/sse")
            print()
            print("🧹 Auto-Cleanup Settings:")
            print("   • Sessions expire: 30 minutes after last activity")
            print("   • Check interval: Every 5 minutes")
            print("   • Orphaned files: Cleaned after 1 hour")
            print()
            print("📊 Available MCP Tools:")
            print("   • upload_temp_file_tool    - Upload files with deduplication")
            print("   • load_dataframe_tool      - Load CSVs, Excel, JSON as DataFrames")
            print("   • run_pandas_code_tool     - Execute pandas operations")
            print("   • create_chart_tool        - Generate interactive visualizations")
            print("   • read_metadata_tool       - Extract comprehensive metadata")
            print("   • list_dataframes_tool     - See all loaded DataFrames")
            print("   ... and 12 more tools")
            print()
            print("Press Ctrl+C to stop the server")
            print("=" * 60)
            
            # Run with SSE transport AND custom FastAPI app
            mcp.run(
                transport="sse",
                host=SERVER_HOST,
                port=SERVER_PORT
            )
        else:
            # Default to stdio transport
            print("Running in stdio mode")
            print("Auto-cleanup: Sessions expire after 30 minutes")
            print("Connect via stdio transport")
            print()
            mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
        print("\n" + "=" * 60)
        print("Shutting down server...")
        
        # Clean all sessions on shutdown
        sessions_dir = Path("sessions")
        if sessions_dir.exists():
            try:
                shutil.rmtree(sessions_dir)
                logger.info("Cleaned all session data on shutdown")
                print("✓ All session data cleaned")
            except Exception as e:
                logger.error(f"Error cleaning sessions: {e}")
        
        print("Server stopped")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"Server failed to start: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()