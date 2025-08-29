"""
Bar chart implementation for Pandas MCP Server.
Supports grouped, stacked, and horizontal bar charts.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.charts.base_chart import BaseChart
from core.config import CHART_COLOR_PALETTE

logger = logging.getLogger(__name__)


class BarChart(BaseChart):
    """Bar chart implementation with interactive controls"""
    
    def __init__(self):
        super().__init__("bar", "Bar Chart")
    
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate data for bar chart"""
        if df.empty:
            return False, "DataFrame is empty"
        
        # Check x_column if specified
        if x_column and x_column not in df.columns:
            return False, f"Column '{x_column}' not found in DataFrame"
        
        # Check y_columns if specified
        if y_columns:
            missing = [col for col in y_columns if col not in df.columns]
            if missing:
                return False, f"Columns not found: {missing}"
            
            # Check if y_columns are numeric
            non_numeric = []
            for col in y_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric.append(col)
            if non_numeric:
                return False, f"Non-numeric columns: {non_numeric}"
        else:
            # Need at least one numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return False, "No numeric columns found in DataFrame"
        
        return True, ""
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare data for bar chart"""
        # Handle groupby if specified
        group_by = kwargs.get('group_by')
        aggregate_func = kwargs.get('aggregate', 'mean')
        
        if group_by and group_by in df.columns:
            # Perform groupby aggregation
            if y_columns:
                agg_df = df.groupby(group_by)[y_columns].agg(aggregate_func).reset_index()
            else:
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                agg_df = df.groupby(group_by)[numeric_cols].agg(aggregate_func).reset_index()
                y_columns = numeric_cols
            
            df = agg_df
            x_column = group_by
        
        # Auto-detect x_column if not specified
        if x_column is None:
            # Try to find a suitable categorical column
            non_numeric = df.select_dtypes(exclude=['number']).columns
            if len(non_numeric) > 0:
                x_column = non_numeric[0]
                x_data = df[x_column].astype(str).tolist()
            else:
                # Use index
                x_data = df.index.astype(str).tolist()
                x_column = "Index"
        else:
            x_data = df[x_column].astype(str).tolist()
        
        # Auto-detect y_columns if not specified
        if y_columns is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if x_column in numeric_cols:
                numeric_cols.remove(x_column)
            y_columns = numeric_cols[:5]  # Limit to 5 series for readability
        
        if not y_columns:
            return {"error": "No numeric columns available for bar chart"}
        
        # Limit data points for performance
        max_points = kwargs.get('max_points', 50)
        if len(x_data) > max_points:
            # Sample evenly
            indices = np.linspace(0, len(x_data)-1, max_points, dtype=int)
            x_data = [x_data[i] for i in indices]
            df = df.iloc[indices]
        
        # Prepare datasets
        datasets = []
        
        # Handle single dataset with multiple bars (different colors per bar)
        if len(y_columns) == 1:
            col = y_columns[0]
            if col not in df.columns:
                return {"error": f"Column '{col}' not found"}
            
            # Handle NaN values
            data_values = df[col].fillna(0).tolist()
            
            # Generate different colors for each bar
            bar_colors = [CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)] 
                         for i in range(len(data_values))]
            hover_colors = [color + "FF" for color in bar_colors]
            background_colors = [color + "CC" for color in bar_colors]
            
            dataset = {
                "label": col,
                "data": data_values,
                "backgroundColor": background_colors,  # Array of colors for each bar
                "borderColor": bar_colors,  # Array of border colors
                "hoverBackgroundColor": hover_colors,  # Hover colors
                "borderWidth": 1,
                "hoverBorderWidth": 2
            }
            
            datasets.append(dataset)
        else:
            # Multiple datasets (different series)
            for i, col in enumerate(y_columns):
                if col not in df.columns:
                    continue
                
                color = CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)]
                
                # Handle NaN values
                data_values = df[col].fillna(0).tolist()
                
                dataset = {
                    "label": col,
                    "data": data_values,
                    "backgroundColor": color + "CC",  # Semi-transparent
                    "borderColor": color,
                    "borderWidth": 1,
                    "hoverBackgroundColor": color + "FF",
                    "hoverBorderWidth": 2
                }
                
                datasets.append(dataset)
        
        return {
            "labels": x_data,
            "datasets": datasets,
            "metadata": {
                "x_column": x_column,
                "y_columns": y_columns,
                "total_records": len(df),
                "aggregation": aggregate_func if group_by else None,
                "single_dataset": len(y_columns) == 1  # Flag for different color handling
            }
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for bar chart"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Bar-specific options
        options.update({
            "indexAxis": "x",  # Can be changed to "y" for horizontal bars
            "responsive": True,
            "maintainAspectRatio": False,
            "scales": {
                "x": {
                    "grid": {
                        "display": False
                    },
                    "ticks": {
                        "maxRotation": 45,
                        "minRotation": 0,
                        "autoSkip": True,
                        "maxTicksLimit": 20
                    }
                },
                "y": {
                    "beginAtZero": True,
                    "grid": {
                        "color": "rgba(0, 0, 0, 0.05)"
                    }
                }
            },
            "plugins": {
                **options.get("plugins", {}),
                "legend": {
                    "display": True,
                    "position": "top",
                    "labels": {
                        "usePointStyle": True,
                        "padding": 15
                    }
                },
                "tooltip": {
                    "enabled": True,
                    "mode": "index",
                    "intersect": False,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "titleColor": "#fff",
                    "bodyColor": "#fff",
                    "borderColor": "#ddd",
                    "borderWidth": 1,
                    "cornerRadius": 4,
                    "padding": 10
                }
            },
            "animation": {
                "duration": 1000,
                "easing": "easeOutQuart"
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
        """Generate bar chart specific controls with custom dropdown"""
        base_controls = super()._generate_controls_html(chart_data)
        
        # Add bar-specific controls
        bar_controls = []
        
        # Stacked toggle
        bar_controls.append('''
            <div class="control-group">
                <label>Stacked</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleStacked(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Horizontal toggle
        bar_controls.append('''
            <div class="control-group">
                <label>Horizontal</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleHorizontal(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Bar width control
        bar_controls.append('''
            <div class="control-group">
                <label>Bar Width</label>
                <input type="range" min="0.3" max="1" step="0.1" value="0.8" 
                       onchange="changeBarWidth(this.value)">
            </div>
        ''')
        
        # Sort control using custom dropdown
        sort_dropdown = self._create_custom_dropdown(
            label="Sort",
            options=[
                ("original", "Original"),
                ("asc", "Ascending"),
                ("desc", "Descending")
            ],
            callback="sortBars",
            default_value="original"
        )
        bar_controls.append(sort_dropdown)
        
        return base_controls + '\n' + '\n'.join(bar_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for bar chart"""
        return '''
        // Add custom plugin to display values on bars
        Chart.register({
            id: 'dataLabels',
            afterDatasetsDraw: function(chart) {
                const ctx = chart.ctx;
                ctx.save();
                
                chart.data.datasets.forEach((dataset, datasetIndex) => {
                    const meta = chart.getDatasetMeta(datasetIndex);
                    if (!meta.hidden) {
                        meta.data.forEach((element, index) => {
                            const value = dataset.data[index];
                            if (value !== null && value !== undefined) {
                                ctx.fillStyle = '#333';
                                ctx.font = 'bold 11px Arial';
                                ctx.textAlign = 'center';
                                ctx.textBaseline = 'bottom';
                                
                                let x = element.x;
                                let y = element.y - 5;
                                
                                // Handle horizontal bars
                                if (chart.options.indexAxis === 'y') {
                                    x = element.x + 10;
                                    y = element.y + 3;
                                    ctx.textAlign = 'left';
                                    ctx.textBaseline = 'middle';
                                }
                                
                                ctx.fillText(value.toFixed(1), x, y);
                            }
                        });
                    }
                });
                
                ctx.restore();
            }
        });
        
        // Add staggered animation delay
        myChart.options.animation.delay = function(context) {
            return context.dataIndex * 100;
        };
        
        function toggleStacked(isStacked) {
            myChart.options.scales.x.stacked = isStacked;
            myChart.options.scales.y.stacked = isStacked;
            myChart.update();
        }
        
        function toggleHorizontal(isHorizontal) {
            if (isHorizontal) {
                myChart.options.indexAxis = 'y';
                const xScale = myChart.options.scales.x;
                const yScale = myChart.options.scales.y;
                myChart.options.scales.x = {
                    ...yScale,
                    grid: { display: true, color: "rgba(0, 0, 0, 0.05)" }
                };
                myChart.options.scales.y = {
                    ...xScale,
                    grid: { display: false }
                };
            } else {
                myChart.options.indexAxis = 'x';
                myChart.options.scales.x = {
                    grid: { display: false },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 20
                    }
                };
                myChart.options.scales.y = {
                    beginAtZero: true,
                    grid: { color: "rgba(0, 0, 0, 0.05)" }
                };
            }
            myChart.update();
        }
        
        function changeBarWidth(width) {
            myChart.data.datasets.forEach(dataset => {
                dataset.barPercentage = parseFloat(width);
            });
            myChart.update();
        }
        
        function toggleDataset(index) {
            const meta = myChart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !myChart.data.datasets[index].hidden : null;
            myChart.update();
        }
        
        function sortBars(order) {
            const labels = [...myChart.data.labels];
            const datasets = myChart.data.datasets;
            
            let indices = [];
            for (let i = 0; i < labels.length; i++) {
                indices.push({
                    label: labels[i],
                    value: datasets[0].data[i],
                    index: i
                });
            }
            
            if (order === 'asc') {
                indices.sort((a, b) => a.value - b.value);
            } else if (order === 'desc') {
                indices.sort((a, b) => b.value - a.value);
            } else {
                indices.sort((a, b) => a.index - b.index);
            }
            
            const newLabels = indices.map(item => item.label);
            datasets.forEach(dataset => {
                const newData = indices.map(item => dataset.data[item.index]);
                dataset.data = newData;
                
                if (Array.isArray(dataset.backgroundColor)) {
                    const newBgColors = indices.map(item => dataset.backgroundColor[item.index]);
                    dataset.backgroundColor = newBgColors;
                }
                if (Array.isArray(dataset.borderColor)) {
                    const newBorderColors = indices.map(item => dataset.borderColor[item.index]);
                    dataset.borderColor = newBorderColors;
                }
                if (Array.isArray(dataset.hoverBackgroundColor)) {
                    const newHoverColors = indices.map(item => dataset.hoverBackgroundColor[item.index]);
                    dataset.hoverBackgroundColor = newHoverColors;
                }
            });
            myChart.data.labels = newLabels;
            
            myChart.update();
        }
        '''