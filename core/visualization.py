"""
Visualization orchestrator for Pandas MCP Server.
Manages chart creation using modular chart implementations.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from core.charts.bar_chart import BarChart
from core.charts.line_chart import LineChart
from core.charts.pie_chart import PieChart
from core.charts.scatter_chart import ScatterChart
from core.charts.heatmap_chart import HeatmapChart
from storage.dataframe_manager import get_manager
from utils.validators import InputValidator

logger = logging.getLogger(__name__)


class VisualizationOrchestrator:
    """
    Orchestrates visualization creation by routing requests to appropriate chart implementations.
    """
    
    # Chart type registry
    CHART_REGISTRY = {
        'bar': BarChart,
        'line': LineChart,
        'pie': PieChart,
        'doughnut': PieChart,  # Pie chart handles doughnut too
        'scatter': ScatterChart,
        'bubble': ScatterChart,  # Scatter chart handles bubble too
        'heatmap': HeatmapChart,
        'correlation': HeatmapChart,  # Heatmap handles correlation matrix
    }
    
    def __init__(self):
        self.df_manager = get_manager()
        self.input_validator = InputValidator()
        self.chart_instances = {}
        
        # Pre-instantiate chart objects for reuse
        self._initialize_charts()
    
    def _initialize_charts(self):
        """Initialize chart instances"""
        self.chart_instances = {
            'bar': BarChart(),
            'line': LineChart(),
            'pie': PieChart(),
            'scatter': ScatterChart(),
            'heatmap': HeatmapChart()
        }
    
    def create_visualization(
        self,
        df_name: str,
        chart_type: str,
        x_column: Optional[str] = None,
        y_columns: Optional[Union[str, List[str]]] = None,
        session_id: str = "default",
        **options
    ) -> Dict[str, Any]:
        """
        Create a visualization from a DataFrame.
        
        This is the main entry point for visualization creation.
        
        Args:
            df_name: Name of the DataFrame to visualize
            chart_type: Type of chart to create
            x_column: Column for x-axis/labels
            y_columns: Column(s) for y-axis/values (can be string or list)
            session_id: Session identifier
            **options: Additional chart-specific options
            
        Returns:
            Dictionary with success status and chart information
        """
        logger.info(f"Creating {chart_type} visualization for DataFrame '{df_name}'")
        
        # Validate inputs
        is_valid, clean_df_name = self.input_validator.validate_dataframe_name(df_name)
        if not is_valid:
            return {"success": False, "error": f"Invalid DataFrame name: {clean_df_name}"}
        
        is_valid, clean_session_id = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            return {"success": False, "error": f"Invalid session ID: {clean_session_id}"}
        
        # Normalize chart type
        chart_type = chart_type.lower()
        
        # Check if chart type is supported
        if chart_type not in self.CHART_REGISTRY:
            supported = list(self.CHART_REGISTRY.keys())
            return {
                "success": False,
                "error": f"Unsupported chart type: {chart_type}. Supported types: {', '.join(supported)}"
            }
        
        # Get DataFrame
        df = self.df_manager.get_dataframe(clean_df_name, clean_session_id)
        if df is None:
            return {
                "success": False,
                "error": f"DataFrame '{clean_df_name}' not found in session '{clean_session_id}'"
            }
        
        # Normalize y_columns to list
        if isinstance(y_columns, str):
            y_columns = [y_columns]
        
        # Get appropriate chart instance
        if chart_type in ['bar', 'line', 'pie', 'scatter', 'heatmap']:
            chart = self.chart_instances[chart_type]
        elif chart_type == 'doughnut':
            chart = self.chart_instances['pie']
            options['is_doughnut'] = True
        elif chart_type == 'bubble':
            chart = self.chart_instances['scatter']
            options['is_bubble'] = True
        elif chart_type == 'correlation':
            chart = self.chart_instances['heatmap']
            options['heatmap_type'] = 'correlation'
        else:
            return {"success": False, "error": f"Chart type '{chart_type}' not properly configured"}
        
        try:
            # Validate data for the chart type
            is_valid, error_msg = chart.validate_data(df, x_column, y_columns, **options)
            if not is_valid:
                return {"success": False, "error": error_msg}
            
            # Prepare data
            chart_data = chart.prepare_data(df, x_column, y_columns, **options)
            if "error" in chart_data:
                return {"success": False, "error": chart_data["error"]}
            
            # Generate HTML
            title = options.pop('title', None) or f"{chart_type.title()} - {clean_df_name}"
            width = options.pop('width', None)
            height = options.pop('height', None)
            
            html_content = chart.generate_html(
                chart_data=chart_data,
                title=title,
                width=width,
                height=height,
                custom_options=options.get('chart_options', {})
            )
            
            # Save chart
            filepath = chart.save_chart(html_content, clean_df_name, suffix=chart_type)
            
            logger.info(f"Successfully created {chart_type} chart: {filepath}")
            
            return {
                "success": True,
                "message": f"{chart_type.title()} chart created successfully",
                "filepath": str(filepath),
                "filename": filepath.name,
                "chart_type": chart_type,
                "data_shape": df.shape,
                "metadata": chart_data.get("metadata", {}),
                "preview_url": f"file://{filepath.absolute()}"
            }
            
        except Exception as e:
            logger.error(f"Failed to create {chart_type} chart: {e}", exc_info=True)
            return {"success": False, "error": f"Failed to create chart: {str(e)}"}
    
    def suggest_charts(self, df_name: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Suggest appropriate chart types based on DataFrame characteristics.
        
        Args:
            df_name: Name of the DataFrame
            session_id: Session identifier
            
        Returns:
            Dictionary with chart suggestions
        """
        # Get DataFrame
        df = self.df_manager.get_dataframe(df_name, session_id)
        if df is None:
            return {
                "success": False,
                "error": f"DataFrame '{df_name}' not found"
            }
        
        suggestions = []
        
        # Analyze DataFrame
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Bar chart - good for categorical + numeric
        if categorical_cols and numeric_cols:
            suggestions.append({
                "chart_type": "bar",
                "reason": "Good for comparing categories",
                "suggested_config": {
                    "x_column": categorical_cols[0],
                    "y_columns": numeric_cols[:3]
                }
            })
        
        # Line chart - good for time series or continuous data
        if datetime_cols and numeric_cols:
            suggestions.append({
                "chart_type": "line",
                "reason": "Perfect for time series data",
                "suggested_config": {
                    "x_column": datetime_cols[0],
                    "y_columns": numeric_cols[:3]
                }
            })
        elif len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "line",
                "reason": "Good for showing trends",
                "suggested_config": {
                    "x_column": None,  # Will use index
                    "y_columns": numeric_cols[:3]
                }
            })
        
        # Pie chart - good for parts of a whole
        if categorical_cols and numeric_cols:
            # Check if numeric column has positive values
            value_col = numeric_cols[0]
            if (df[value_col] >= 0).all():
                suggestions.append({
                    "chart_type": "pie",
                    "reason": "Shows proportions of categories",
                    "suggested_config": {
                        "x_column": categorical_cols[0],
                        "y_columns": [value_col]
                    }
                })
        
        # Scatter plot - good for correlations
        if len(numeric_cols) >= 2:
            suggestions.append({
                "chart_type": "scatter",
                "reason": "Reveals relationships between variables",
                "suggested_config": {
                    "x_column": numeric_cols[0],
                    "y_columns": numeric_cols[1:3]
                }
            })
        
        # Heatmap - good for correlation matrix
        if len(numeric_cols) >= 3:
            suggestions.append({
                "chart_type": "correlation",
                "reason": "Shows correlations between all numeric columns",
                "suggested_config": {
                    "heatmap_type": "correlation",
                    "y_columns": numeric_cols
                }
            })
        
        return {
            "success": True,
            "dataframe": df_name,
            "shape": df.shape,
            "column_types": {
                "numeric": len(numeric_cols),
                "categorical": len(categorical_cols),
                "datetime": len(datetime_cols)
            },
            "suggestions": suggestions[:4],  # Limit to top 4 suggestions
            "total_suggestions": len(suggestions)
        }
    
    def get_supported_charts(self) -> Dict[str, Any]:
        """
        Get information about all supported chart types.
        
        Returns:
            Dictionary with chart type information
        """
        chart_info = {}
        
        for chart_type, chart_class in self.CHART_REGISTRY.items():
            if chart_type in ['doughnut', 'bubble', 'correlation']:
                # These are aliases
                continue
            
            instance = self.chart_instances.get(chart_type)
            if instance:
                chart_info[chart_type] = {
                    "name": instance.chart_name,
                    "description": self._get_chart_description(chart_type),
                    "required_data": self._get_required_data(chart_type),
                    "options": self._get_chart_options(chart_type)
                }
        
        return {
            "supported_charts": chart_info,
            "aliases": {
                "doughnut": "pie",
                "bubble": "scatter",
                "correlation": "heatmap"
            }
        }
    
    def _get_chart_description(self, chart_type: str) -> str:
        """Get description for a chart type"""
        descriptions = {
            "bar": "Compare values across categories. Supports grouped and stacked bars.",
            "line": "Show trends over time or continuous data. Ideal for time series.",
            "pie": "Display proportions of a whole. Can also create doughnut charts.",
            "scatter": "Reveal relationships between variables. Supports bubble charts with size dimension.",
            "heatmap": "Visualize matrix data or correlations with color intensity."
        }
        return descriptions.get(chart_type, "")
    
    def _get_required_data(self, chart_type: str) -> Dict[str, str]:
        """Get required data description for a chart type"""
        requirements = {
            "bar": {
                "x_column": "Categorical or discrete values (optional)",
                "y_columns": "One or more numeric columns"
            },
            "line": {
                "x_column": "Time or continuous values (optional)",
                "y_columns": "One or more numeric columns"
            },
            "pie": {
                "x_column": "Category labels",
                "y_columns": "Single numeric column for values"
            },
            "scatter": {
                "x_column": "Numeric values for x-axis",
                "y_columns": "One or more numeric columns for y-axis"
            },
            "heatmap": {
                "x_column": "Column for pivot (optional)",
                "y_columns": "Columns to include in correlation/pivot"
            }
        }
        return requirements.get(chart_type, {})
    
    def _get_chart_options(self, chart_type: str) -> List[Dict[str, str]]:
        """Get available options for a chart type"""
        common_options = [
            {"name": "title", "type": "string", "description": "Chart title"},
            {"name": "width", "type": "integer", "description": "Chart width in pixels"},
            {"name": "height", "type": "integer", "description": "Chart height in pixels"}
        ]
        
        specific_options = {
            "bar": [
                {"name": "group_by", "type": "string", "description": "Column to group by"},
                {"name": "aggregate", "type": "string", "description": "Aggregation function (mean, sum, etc.)"},
                {"name": "max_points", "type": "integer", "description": "Maximum number of bars"}
            ],
            "line": [
                {"name": "tension", "type": "float", "description": "Line curve tension (0-0.5)"},
                {"name": "fill", "type": "boolean", "description": "Fill area under line"},
                {"name": "max_points", "type": "integer", "description": "Maximum data points"}
            ],
            "pie": [
                {"name": "max_slices", "type": "integer", "description": "Maximum number of slices"},
                {"name": "aggregate_by", "type": "string", "description": "Column to aggregate by"},
                {"name": "aggregate_func", "type": "string", "description": "Aggregation function"}
            ],
            "scatter": [
                {"name": "size_column", "type": "string", "description": "Column for bubble size"},
                {"name": "show_trend", "type": "boolean", "description": "Show trend lines"},
                {"name": "max_points", "type": "integer", "description": "Maximum points to display"}
            ],
            "heatmap": [
                {"name": "heatmap_type", "type": "string", "description": "Type: correlation, pivot, or matrix"},
                {"name": "color_scheme", "type": "string", "description": "Color scheme: viridis, coolwarm, grayscale"},
                {"name": "value_column", "type": "string", "description": "Column for pivot values"}
            ]
        }
        
        return common_options + specific_options.get(chart_type, [])


# Global instance
_orchestrator_instance = None

def get_orchestrator() -> VisualizationOrchestrator:
    """Get the global VisualizationOrchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = VisualizationOrchestrator()
    return _orchestrator_instance