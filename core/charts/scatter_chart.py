"""
Scatter chart implementation for Pandas MCP Server.
Supports bubble charts, multiple series, and trend lines.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.charts.base_chart import BaseChart
from core.config import CHART_COLOR_PALETTE

# Try to import scipy for trend lines, but make it optional
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logging.warning("scipy not installed - trend lines will not be available in scatter plots")

logger = logging.getLogger(__name__)


class ScatterChart(BaseChart):
    """Scatter plot implementation with regression support"""
    
    def __init__(self):
        super().__init__("scatter", "Scatter Plot")
    
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate data for scatter chart"""
        if df.empty:
            return False, "DataFrame is empty"
        
        # Need at least 2 numeric columns for scatter plot
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) < 2:
            return False, "Scatter plot requires at least 2 numeric columns"
        
        # Check x_column if specified
        if x_column:
            if x_column not in df.columns:
                return False, f"X column '{x_column}' not found"
            if not pd.api.types.is_numeric_dtype(df[x_column]):
                return False, f"X column '{x_column}' must be numeric"
        
        # Check y_columns if specified
        if y_columns:
            for col in y_columns:
                if col not in df.columns:
                    return False, f"Y column '{col}' not found"
                if not pd.api.types.is_numeric_dtype(df[col]):
                    return False, f"Y column '{col}' must be numeric"
        
        return True, ""
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare data for scatter chart"""
        # Get size column for bubble chart
        size_column = kwargs.get('size_column')
        color_column = kwargs.get('color_column')
        show_trend = kwargs.get('show_trend', False)
        
        # Auto-detect columns if not specified
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if x_column is None:
            if len(numeric_cols) >= 1:
                x_column = numeric_cols[0]
            else:
                return {"error": "No numeric column found for X axis"}
        
        if y_columns is None:
            available = [col for col in numeric_cols if col != x_column]
            if available:
                y_columns = available[:3]  # Limit to 3 series
            else:
                return {"error": "No numeric columns found for Y axis"}
        
        # Limit data points for performance
        max_points = kwargs.get('max_points', 1000)
        if len(df) > max_points:
            df = df.sample(n=max_points, random_state=42)
        
        # Prepare datasets
        datasets = []
        trend_lines = []
        
        for i, y_col in enumerate(y_columns):
            if y_col not in df.columns:
                continue
            
            color = CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)]
            
            # Remove NaN values
            clean_data = df[[x_column, y_col]].dropna()
            
            if len(clean_data) == 0:
                continue
            
            # Prepare point data
            point_data = []
            for _, row in clean_data.iterrows():
                point = {
                    "x": float(row[x_column]),
                    "y": float(row[y_col])
                }
                
                # Add size for bubble chart
                if size_column and size_column in df.columns:
                    size_value = row[size_column] if size_column in row else 5
                    point["r"] = max(3, min(30, float(size_value) / df[size_column].max() * 20))
                
                point_data.append(point)
            
            dataset = {
                "label": y_col,
                "data": point_data,
                "backgroundColor": color + "80",
                "borderColor": color,
                "borderWidth": 1,
                "pointRadius": 4,
                "pointHoverRadius": 6,
                "showLine": False  # Scatter points only
            }
            
            datasets.append(dataset)
            
            # Calculate trend line if requested
            if show_trend and len(clean_data) > 1 and HAS_SCIPY:
                x_values = clean_data[x_column].values
                y_values = clean_data[y_col].values
                
                try:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                    
                    # Generate trend line points
                    x_min, x_max = x_values.min(), x_values.max()
                    trend_x = [x_min, x_max]
                    trend_y = [slope * x_min + intercept, slope * x_max + intercept]
                    
                    trend_dataset = {
                        "label": f"{y_col} (trend)",
                        "data": [
                            {"x": trend_x[0], "y": trend_y[0]},
                            {"x": trend_x[1], "y": trend_y[1]}
                        ],
                        "type": "line",
                        "fill": False,
                        "borderColor": color,
                        "borderWidth": 2,
                        "borderDash": [5, 5],
                        "pointRadius": 0,
                        "showLine": True,
                        "tension": 0
                    }
                    
                    trend_lines.append({
                        "dataset": trend_dataset,
                        "equation": f"y = {slope:.3f}x + {intercept:.3f}",
                        "r_squared": r_value ** 2
                    })
                except Exception as e:
                    logger.warning(f"Failed to calculate trend line: {e}")
            elif show_trend and not HAS_SCIPY:
                logger.info("Trend lines requested but scipy not available")
        
        # Add trend lines to datasets
        for trend in trend_lines:
            datasets.append(trend["dataset"])
        
        # Calculate correlations
        correlations = {}
        if len(y_columns) > 0:
            for y_col in y_columns:
                if y_col in df.columns:
                    clean = df[[x_column, y_col]].dropna()
                    if len(clean) > 1:
                        corr = clean[x_column].corr(clean[y_col])
                        correlations[y_col] = corr
        
        return {
            "datasets": datasets,
            "metadata": {
                "x_column": x_column,
                "y_columns": y_columns,
                "total_points": len(df),
                "correlations": correlations,
                "trend_lines": [{"equation": t["equation"], "r_squared": t["r_squared"]} for t in trend_lines],
                "is_bubble": size_column is not None
            }
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for scatter chart"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Scatter-specific options
        options.update({
            "scales": {
                "x": {
                    "type": "linear",
                    "position": "bottom",
                    "grid": {
                        "color": "rgba(0, 0, 0, 0.05)"
                    },
                    "title": {
                        "display": True,
                        "text": "X Axis"
                    }
                },
                "y": {
                    "type": "linear",
                    "grid": {
                        "color": "rgba(0, 0, 0, 0.05)"
                    },
                    "title": {
                        "display": True,
                        "text": "Y Axis"
                    }
                }
            },
            "plugins": {
                **options.get("plugins", {}),
                "legend": {
                    "display": True,
                    "position": "top",
                    "labels": {
                        "usePointStyle": True
                        # filter will be set in JavaScript
                    }
                },
                "tooltip": {
                    "enabled": True,
                    "mode": "point",
                    "intersect": True,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "callbacks": {}  # Will be set in JavaScript
                },
                "zoom": {
                    "enabled": False,  # Could be enabled with plugin
                    "mode": "xy"
                }
            },
            "interaction": {
                "mode": "point",
                "intersect": True
            }
        })
        
        return options
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate scatter chart specific controls"""
        base_controls = super()._generate_controls_html(chart_data)
        
        # Add scatter-specific controls
        scatter_controls = []
        
        # Point size control
        scatter_controls.append('''
            <div class="control-group">
                <label>Point Size</label>
                <input type="range" min="1" max="10" step="1" value="4" 
                       onchange="changePointSize(this.value)">
            </div>
        ''')
        
        # Show trend lines
        scatter_controls.append('''
            <div class="control-group">
                <label>Trend Lines</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleTrendLines(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Grid toggle
        scatter_controls.append('''
            <div class="control-group">
                <label>Grid</label>
                <label class="switch">
                    <input type="checkbox" checked onchange="toggleGrid(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Logarithmic scale toggle
        scatter_controls.append('''
            <div class="control-group">
                <label>Log Scale</label>
                <select onchange="changeScale(this.value)">
                    <option value="linear">Linear</option>
                    <option value="logarithmic">Logarithmic</option>
                </select>
            </div>
        ''')
        
        # Add correlation info if available
        if "correlations" in chart_data.get("metadata", {}):
            corr_html = '<div class="info-panel" style="margin-top: 10px;"><strong>Correlations:</strong><br>'
            for col, corr in chart_data["metadata"]["correlations"].items():
                corr_html += f'{col}: {corr:.3f}<br>'
            corr_html += '</div>'
            scatter_controls.append(corr_html)
        
        return base_controls + '\n' + '\n'.join(scatter_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for scatter chart"""
        return '''
        // Enhanced tooltips with correlation info
        myChart.options.plugins.tooltip.callbacks.label = function(context) {
            let label = context.dataset.label || '';
            if (label) {
                label += ': ';
            }
            label += '(' + context.parsed.x.toFixed(2) + ', ' + context.parsed.y.toFixed(2) + ')';
            if (context.raw.r) {
                label += ', size: ' + context.raw.r.toFixed(1);
            }
            return label;
        };
        
        // Filter out trend lines from legend if hidden
        let showTrends = false;
        myChart.options.plugins.legend.labels.filter = function(item, chart) {
            if (!showTrends && item.text.includes('trend')) {
                return false;
            }
            return true;
        };
        
        function changePointSize(size) {
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.label.includes('trend')) {
                    dataset.pointRadius = parseInt(size);
                    dataset.pointHoverRadius = parseInt(size) + 2;
                }
            });
            myChart.update();
        }
        
        function toggleTrendLines(show) {
            showTrends = show;
            myChart.data.datasets.forEach(dataset => {
                if (dataset.label.includes('trend')) {
                    dataset.hidden = !show;
                }
            });
            myChart.options.plugins.legend.labels.filter = function(item, chart) {
                if (!showTrends && item.text.includes('trend')) {
                    return false;
                }
                return true;
            };
            myChart.update();
        }
        
        function toggleGrid(show) {
            myChart.options.scales.x.grid.display = show;
            myChart.options.scales.y.grid.display = show;
            myChart.update();
        }
        
        function changeScale(scaleType) {
            myChart.options.scales.x.type = scaleType;
            myChart.options.scales.y.type = scaleType;
            myChart.update();
        }
        
        // Quadrant highlighting
        function addQuadrants() {
            const xScale = myChart.scales.x;
            const yScale = myChart.scales.y;
            const xMid = (xScale.max + xScale.min) / 2;
            const yMid = (yScale.max + yScale.min) / 2;
            
            // Add annotation plugin config for quadrants
            myChart.options.plugins.annotation = {
                annotations: {
                    line1: {
                        type: 'line',
                        xMin: xMid,
                        xMax: xMid,
                        borderColor: 'rgba(128, 128, 128, 0.3)',
                        borderWidth: 1,
                        borderDash: [5, 5]
                    },
                    line2: {
                        type: 'line',
                        yMin: yMid,
                        yMax: yMid,
                        borderColor: 'rgba(128, 128, 128, 0.3)',
                        borderWidth: 1,
                        borderDash: [5, 5]
                    }
                }
            };
            myChart.update();
        }
        
        // Highlight outliers
        function highlightOutliers() {
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.label.includes('trend')) {
                    const values = dataset.data.map(p => p.y);
                    const mean = values.reduce((a, b) => a + b, 0) / values.length;
                    const std = Math.sqrt(values.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / values.length);
                    
                    dataset.pointBackgroundColor = dataset.data.map(point => {
                        if (Math.abs(point.y - mean) > 2 * std) {
                            return '#ff0000';  // Red for outliers
                        }
                        return dataset.backgroundColor;
                    });
                }
            });
            myChart.update();
        }
        '''