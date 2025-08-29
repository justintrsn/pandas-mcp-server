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
                    }
                },
                "tooltip": {
                    "enabled": True,
                    "mode": "point",
                    "intersect": True,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "titleColor": "#fff",
                    "bodyColor": "#fff",
                    "borderColor": "#ddd",
                    "borderWidth": 1,
                    "cornerRadius": 4,
                    "padding": 10
                }
            },
            "interaction": {
                "mode": "point",
                "intersect": True
            },
            "animation": {
                "duration": 1200,
                "easing": "easeOutBounce"
            },
            "transitions": {
                "active": {
                    "animation": {
                        "duration": 400
                    }
                }
            }
        })
        
        return options

    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate scatter chart specific controls with custom dropdown"""
        base_controls = super()._generate_controls_html(chart_data)
        
        scatter_controls = []
        
        # Point size control
        scatter_controls.append('''
            <div class="control-group">
                <label>Point Size</label>
                <input type="range" min="1" max="15" step="1" value="4" 
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
        
        # Show statistics
        scatter_controls.append('''
            <div class="control-group">
                <label>Show Stats</label>
                <label class="switch">
                    <input type="checkbox" checked onchange="toggleStats(this.checked)">
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
        
        # Scale type using custom dropdown
        scale_dropdown = self._create_custom_dropdown(
            label="Scale",
            options=[
                ("linear", "Linear"),
                ("logarithmic", "Logarithmic")
            ],
            callback="changeScale",
            default_value="linear"
        )
        scatter_controls.append(scale_dropdown)
        
        # Point shape using custom dropdown
        shape_dropdown = self._create_custom_dropdown(
            label="Point Shape",
            options=[
                ("circle", "Circle"),
                ("triangle", "Triangle"),
                ("rect", "Square"),
                ("rectRot", "Diamond"),
                ("cross", "Cross"),
                ("star", "Star")
            ],
            callback="changePointStyle",
            default_value="circle"
        )
        scatter_controls.append(shape_dropdown)
        
        return base_controls + '\n' + '\n'.join(scatter_controls)

    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for scatter chart"""
        return '''
        // Add staggered point animation
        myChart.options.animation.delay = function(context) {
            return Math.random() * 1000;
        };
        
        // Store metadata for statistics display
        myChart.correlationData = chartData.metadata.correlations || {};
        myChart.trendData = chartData.metadata.trend_lines || [];
        
        // Plugin to display statistics on the chart
        Chart.register({
            id: 'statsDisplay',
            afterDraw: function(chart) {
                if (chart.config.showStats === false) return;
                
                const ctx = chart.ctx;
                ctx.save();
                
                // Background box for statistics
                const padding = 10;
                const lineHeight = 20;
                const startX = chart.width - 200;
                const startY = 60;
                
                const stats = [];
                
                // Add correlation data
                if (chart.correlationData) {
                    for (const [key, value] of Object.entries(chart.correlationData)) {
                        stats.push(`${key}: r = ${value.toFixed(3)}`);
                    }
                }
                
                // Add R-squared from trend lines
                if (chart.trendData && chart.trendData.length > 0) {
                    chart.trendData.forEach((trend, index) => {
                        if (trend.r_squared !== undefined) {
                            stats.push(`R² = ${trend.r_squared.toFixed(3)}`);
                        }
                    });
                }
                
                if (stats.length > 0) {
                    // Draw background
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
                    ctx.strokeStyle = 'rgba(102, 126, 234, 0.5)';
                    ctx.lineWidth = 1;
                    
                    const boxHeight = stats.length * lineHeight + padding * 2;
                    const boxWidth = 180;
                    
                    ctx.fillRect(startX, startY, boxWidth, boxHeight);
                    ctx.strokeRect(startX, startY, boxWidth, boxHeight);
                    
                    // Draw text
                    ctx.fillStyle = '#333';
                    ctx.font = '12px Arial';
                    ctx.textAlign = 'left';
                    
                    stats.forEach((stat, index) => {
                        ctx.fillText(stat, startX + padding, startY + padding + (index + 0.8) * lineHeight);
                    });
                }
                
                ctx.restore();
            }
        });
        
        // Initialize stats display
        myChart.config.showStats = true;
        
        function changePointSize(size) {
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.label || !dataset.label.includes('(trend)')) {
                    dataset.pointRadius = parseInt(size);
                    dataset.pointHoverRadius = parseInt(size) + 2;
                }
            });
            myChart.update();
        }
        
        function toggleTrendLines(show) {
            myChart.data.datasets.forEach((dataset, index) => {
                if (dataset.label && dataset.label.includes('(trend)')) {
                    const meta = myChart.getDatasetMeta(index);
                    meta.hidden = !show;
                }
            });
            myChart.update();
        }
        
        function toggleStats(show) {
            myChart.config.showStats = show;
            myChart.update();
        }
        
        function toggleGrid(show) {
            myChart.options.scales.x.grid.display = show;
            myChart.options.scales.y.grid.display = show;
            myChart.update();
        }
        
        function changeScale(scaleType) {
            const isLog = scaleType === 'logarithmic';
            
            // Store original data on first switch
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.originalData) {
                    dataset.originalData = [...dataset.data];
                }
            });
            
            if (isLog) {
                // For logarithmic scale, we need to ensure all values are positive
                myChart.options.scales.x.type = 'logarithmic';
                myChart.options.scales.y.type = 'logarithmic';
                
                // Set proper min values for log scale
                myChart.options.scales.x.min = undefined;
                myChart.options.scales.y.min = undefined;
                
                // Filter data points
                myChart.data.datasets.forEach(dataset => {
                    if (!dataset.label || !dataset.label.includes('(trend)')) {
                        // For scatter points, filter out non-positive values
                        dataset.data = dataset.originalData.filter(point => 
                            point.x > 0 && point.y > 0
                        );
                    } else {
                        // For trend lines, recalculate with positive values only
                        // Find the corresponding scatter dataset
                        const scatterLabel = dataset.label.replace(' (trend)', '');
                        const scatterDataset = myChart.data.datasets.find(d => 
                            d.label === scatterLabel
                        );
                        
                        if (scatterDataset && scatterDataset.data.length > 1) {
                            // Get positive values only
                            const validPoints = scatterDataset.data.filter(p => p.x > 0 && p.y > 0);
                            if (validPoints.length > 1) {
                                // Calculate log-transformed trend line
                                const xValues = validPoints.map(p => Math.log10(p.x));
                                const yValues = validPoints.map(p => Math.log10(p.y));
                                
                                // Simple linear regression on log values
                                const n = xValues.length;
                                const sumX = xValues.reduce((a, b) => a + b, 0);
                                const sumY = yValues.reduce((a, b) => a + b, 0);
                                const sumXY = xValues.reduce((sum, x, i) => sum + x * yValues[i], 0);
                                const sumX2 = xValues.reduce((sum, x) => sum + x * x, 0);
                                
                                const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
                                const intercept = (sumY - slope * sumX) / n;
                                
                                // Generate trend line in log space
                                const xMin = Math.min(...validPoints.map(p => p.x));
                                const xMax = Math.max(...validPoints.map(p => p.x));
                                
                                dataset.data = [
                                    {x: xMin, y: Math.pow(10, slope * Math.log10(xMin) + intercept)},
                                    {x: xMax, y: Math.pow(10, slope * Math.log10(xMax) + intercept)}
                                ];
                            } else {
                                dataset.data = [];
                            }
                        }
                    }
                });
            } else {
                // Reset to linear scale
                myChart.options.scales.x.type = 'linear';
                myChart.options.scales.y.type = 'linear';
                
                delete myChart.options.scales.x.min;
                delete myChart.options.scales.y.min;
                
                // Restore original data
                myChart.data.datasets.forEach(dataset => {
                    if (dataset.originalData) {
                        dataset.data = [...dataset.originalData];
                    }
                });
            }
            
            myChart.update();
        }
        
        function changePointStyle(style) {
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.label || !dataset.label.includes('(trend)')) {
                    dataset.pointStyle = style;
                }
            });
            myChart.update();
        }
        
        function toggleDataset(index) {
            const meta = myChart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !myChart.data.datasets[index].hidden : null;
            myChart.update();
        }
        '''