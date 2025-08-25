"""
Core functionality for Pandas MCP Server.
"""

# Import main components
from core.config import get_config
from core.metadata import MetadataExtractor
from core.execution import PandasExecutor
from core.data_loader import DataLoader
from core.data_types import DataTypeDetector
from core.visualization import VisualizationOrchestrator

# Import tool functions for easy access
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
    # Visualization tools
    create_chart,
    suggest_charts,
    get_chart_types,
    create_correlation_heatmap,
    create_time_series_chart
)

# Import chart classes for advanced usage
from core.charts import (
    BaseChart,
    BarChart,
    LineChart,
    PieChart,
    ScatterChart,
    HeatmapChart
)

__all__ = [
    # Classes
    'MetadataExtractor',
    'PandasExecutor',
    'DataLoader',
    'DataTypeDetector',
    'VisualizationOrchestrator',
    
    # Chart Classes
    'BaseChart',
    'BarChart',
    'LineChart',
    'PieChart',
    'ScatterChart',
    'HeatmapChart',
    
    # Core Functions
    'get_config',
    'read_metadata',
    'run_pandas_code',
    'load_dataframe',
    'list_dataframes',
    'get_dataframe_info',
    'validate_pandas_code',
    'get_execution_context',
    'preview_file',
    'get_supported_formats',
    'clear_session',
    'get_session_info',
    'get_server_info',
    
    # Visualization Functions
    'create_chart',
    'suggest_charts',
    'get_chart_types',
    'create_correlation_heatmap',
    'create_time_series_chart'
]