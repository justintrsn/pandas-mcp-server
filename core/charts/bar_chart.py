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
                "aggregation": aggregate_func if group_by else None
            }
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for bar chart"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Bar-specific options
        options.update({
            "indexAxis": "x",  # Can be changed to "y" for horizontal bars
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
            }
        })
        
        return options
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate bar chart specific controls"""
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
        
        return base_controls + '\n' + '\n'.join(bar_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for bar chart"""
        return '''
        function toggleStacked(isStacked) {
            myChart.options.scales.x.stacked = isStacked;
            myChart.options.scales.y.stacked = isStacked;
            myChart.update();
        }
        
        function toggleHorizontal(isHorizontal) {
            myChart.options.indexAxis = isHorizontal ? 'y' : 'x';
            myChart.update();
        }
        
        function changeBarWidth(width) {
            myChart.data.datasets.forEach(dataset => {
                dataset.barPercentage = parseFloat(width);
            });
            myChart.update();
        }
        
        // Add dataset visibility toggles
        function toggleDataset(index) {
            const meta = myChart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !myChart.data.datasets[index].hidden : null;
            myChart.update();
        }
        
        // Sort bars
        function sortBars(order) {
            const labels = myChart.data.labels;
            const datasets = myChart.data.datasets;
            
            // Create array of indices and values for sorting
            let indices = [];
            for (let i = 0; i < labels.length; i++) {
                indices.push({
                    label: labels[i],
                    value: datasets[0].data[i],
                    index: i
                });
            }
            
            // Sort based on order
            if (order === 'asc') {
                indices.sort((a, b) => a.value - b.value);
            } else if (order === 'desc') {
                indices.sort((a, b) => b.value - a.value);
            } else {
                // Original order
                indices.sort((a, b) => a.index - b.index);
            }
            
            // Reorder data
            const newLabels = indices.map(item => item.label);
            datasets.forEach(dataset => {
                const newData = indices.map(item => dataset.data[item.index]);
                dataset.data = newData;
            });
            myChart.data.labels = newLabels;
            
            myChart.update();
        }
        '''