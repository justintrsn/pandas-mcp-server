#!/usr/bin/env python3
"""
Pandas MCP Server - THIN LAYER
This is just the MCP interface layer that calls functions from core.tools
"""

import sys
import os
import logging
from typing import Dict, Any, Optional
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
    get_server_info
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
# MCP TOOL DEFINITIONS - THIN WRAPPERS
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
        filepath: Path to the data file to analyze
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
    
    Args:
        filepath: Path to the file to load
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
        filepath: Path to the file
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
    Clear all data in a session.
    
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
# MAIN SERVER FUNCTION
# ============================================================================

def main():
    """Main entry point - just starts the MCP server"""
    try:
        # Print startup banner
        print("=" * 60)
        print(f" {SERVER_NAME} v{SERVER_VERSION}")
        print(f" {SERVER_DESCRIPTION}")
        print("=" * 60)
        print()
        
        logger.info(f"Starting {SERVER_NAME} v{SERVER_VERSION}")
        logger.info(f"Server configuration: {SERVER_TRANSPORT} on {SERVER_HOST}:{SERVER_PORT}")
        
        # Log available tools
        logger.info("Available MCP tools:")
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
        
        # Start server based on transport
        if SERVER_TRANSPORT == "sse":
            # SSE transport for HTTP
            print(f"Server running at http://{SERVER_HOST}:{SERVER_PORT}")
            print("Connect your MCP client to this URL")
            print()
            print("Available tools:")
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