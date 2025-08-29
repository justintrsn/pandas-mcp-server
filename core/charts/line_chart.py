"""
Line chart implementation for Pandas MCP Server.
Supports multi-series, area fills, and time series data.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from core.charts.base_chart import BaseChart
from core.config import CHART_COLOR_PALETTE

logger = logging.getLogger(__name__)


class LineChart(BaseChart):
    """Line chart implementation with time series support"""
    
    def __init__(self):
        super().__init__("line", "Line Chart")
    
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate data for line chart"""
        if df.empty:
            return False, "DataFrame is empty"
        
        if len(df) < 2:
            return False, "Line chart requires at least 2 data points"
        
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
        """Prepare data for line chart"""
        # Check if time series
        is_time_series = False
        time_column = None
        
        # Auto-detect time column if not specified
        if x_column is None:
            # Look for datetime columns
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                x_column = datetime_cols[0]
                is_time_series = True
                time_column = x_column
            else:
                # Check if index is datetime
                if pd.api.types.is_datetime64_any_dtype(df.index):
                    x_data = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
                    x_column = df.index.name or "Time"
                    is_time_series = True
                else:
                    # Use first non-numeric column or index
                    non_numeric = df.select_dtypes(exclude=['number']).columns
                    if len(non_numeric) > 0:
                        x_column = non_numeric[0]
                    else:
                        x_data = list(range(len(df)))
                        x_column = "Index"
        
        # Prepare x_data
        if x_column in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[x_column]):
                x_data = pd.to_datetime(df[x_column]).dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
                is_time_series = True
                time_column = x_column
            else:
                x_data = df[x_column].astype(str).tolist()
        elif 'x_data' not in locals():
            x_data = list(range(len(df)))
        
        # Auto-detect y_columns if not specified
        if y_columns is None:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if x_column in numeric_cols:
                numeric_cols.remove(x_column)
            y_columns = numeric_cols[:5]  # Limit to 5 series
        
        if not y_columns:
            return {"error": "No numeric columns available for line chart"}
        
        # Resample if too many points
        max_points = kwargs.get('max_points', 200)
        if len(df) > max_points:
            if is_time_series and time_column:
                # Resample time series data
                df_resampled = self._resample_time_series(df, time_column, y_columns, max_points)
                x_data = pd.to_datetime(df_resampled.index).strftime('%Y-%m-%d %H:%M:%S').tolist()
                df = df_resampled
            else:
                # Sample evenly
                indices = np.linspace(0, len(df)-1, max_points, dtype=int)
                x_data = [x_data[i] for i in indices]
                df = df.iloc[indices]
        
        # Prepare datasets
        datasets = []
        for i, col in enumerate(y_columns):
            if col not in df.columns:
                continue
            
            color = CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)]
            
            # Handle NaN values - interpolate for line charts
            data_series = df[col].copy()
            if data_series.isna().any():
                data_series = data_series.interpolate(method='linear', limit_direction='both')
                data_series = data_series.fillna(method='bfill').fillna(method='ffill')
            
            dataset = {
                "label": col,
                "data": data_series.tolist(),
                "borderColor": color,
                "backgroundColor": color + "20",  # Very transparent for area fill
                "borderWidth": 2,
                "pointRadius": 3,
                "pointHoverRadius": 5,
                "pointBackgroundColor": color,
                "pointBorderColor": "#fff",
                "pointBorderWidth": 1,
                "tension": kwargs.get('tension', 0.2),  # Curve tension
                "fill": kwargs.get('fill', False),  # Area fill
                "borderDash": [],  # For line style
                "stepped": False  # For stepped interpolation
            }
            
            datasets.append(dataset)
        
        return {
            "labels": x_data,
            "datasets": datasets,
            "metadata": {
                "x_column": x_column,
                "y_columns": y_columns,
                "is_time_series": is_time_series,
                "total_records": len(df),
                "resampled": len(df) < len(x_data)
            }
        }
    

    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for line chart"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Line-specific options
        options.update({
            "interaction": {
                "mode": "index",
                "intersect": False
            },
            "scales": {
                "x": {
                    "grid": {
                        "display": True,
                        "color": "rgba(0, 0, 0, 0.05)"
                    },
                    "ticks": {
                        "maxRotation": 45,
                        "minRotation": 0,
                        "autoSkip": True,
                        "maxTicksLimit": 20
                    }
                },
                "y": {
                    "beginAtZero": False,
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
                "easing": "easeInOutQuart"
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
    
    def _resample_time_series(
        self,
        df: pd.DataFrame,
        time_column: str,
        value_columns: List[str],
        target_points: int
    ) -> pd.DataFrame:
        """Resample time series data to target number of points"""
        # Set time column as index
        df_ts = df.set_index(time_column) if time_column in df.columns else df
        
        # Calculate appropriate frequency
        time_range = df_ts.index.max() - df_ts.index.min()
        freq = time_range / target_points
        
        # Determine pandas frequency string
        if freq.days > 365:
            freq_str = f'{int(freq.days/365)}Y'
        elif freq.days > 30:
            freq_str = f'{int(freq.days/30)}M'
        elif freq.days > 7:
            freq_str = f'{int(freq.days/7)}W'
        elif freq.days > 0:
            freq_str = f'{int(freq.days)}D'
        elif freq.seconds > 3600:
            freq_str = f'{int(freq.seconds/3600)}H'
        elif freq.seconds > 60:
            freq_str = f'{int(freq.seconds/60)}T'
        else:
            freq_str = f'{int(freq.seconds)}S'
        
        # Resample
        resampled = df_ts[value_columns].resample(freq_str).mean()
        
        return resampled
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate line chart specific controls with custom dropdowns"""
        base_controls = super()._generate_controls_html(chart_data)
        
        # Add line-specific controls
        line_controls = []
        
        # Interpolation mode using custom dropdown
        interpolation_dropdown = self._create_custom_dropdown(
            label="Line Style",
            options=[
                ("smooth", "Smooth"),
                ("linear", "Linear"),
                ("stepped", "Stepped")
            ],
            callback="changeInterpolation",
            default_value="smooth"
        )
        line_controls.append(interpolation_dropdown)
        
        # Line pattern using custom dropdown
        pattern_dropdown = self._create_custom_dropdown(
            label="Line Pattern",
            options=[
                ("solid", "Solid"),
                ("dashed", "Dashed"),
                ("dotted", "Dotted")
            ],
            callback="changeLinePattern",
            default_value="solid"
        )
        line_controls.append(pattern_dropdown)
        
        # Curve tension control (for smooth interpolation)
        line_controls.append('''
            <div class="control-group">
                <label>Smoothness</label>
                <input type="range" min="0" max="0.5" step="0.05" value="0.2" 
                       onchange="changeTension(this.value)">
            </div>
        ''')
        
        # Fill area toggle
        line_controls.append('''
            <div class="control-group">
                <label>Fill Area</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleFill(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Points visibility
        line_controls.append('''
            <div class="control-group">
                <label>Show Points</label>
                <label class="switch">
                    <input type="checkbox" checked onchange="togglePoints(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Y-axis zero toggle
        line_controls.append('''
            <div class="control-group">
                <label>Start at Zero</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleBeginAtZero(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        return base_controls + '\n' + '\n'.join(line_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for line chart"""
        return '''
        // Add progressive animation
        myChart.options.animation.delay = function(context) {
            return context.dataIndex * 50;
        };
        
        function changeInterpolation(mode) {
            myChart.data.datasets.forEach(dataset => {
                switch(mode) {
                    case 'smooth':
                        dataset.tension = 0.3;
                        dataset.stepped = false;
                        break;
                    case 'linear':
                        dataset.tension = 0;
                        dataset.stepped = false;
                        break;
                    case 'stepped':
                        dataset.tension = 0;
                        dataset.stepped = 'before';
                        break;
                }
            });
            
            // Update smoothness slider visibility
            const smoothnessControl = document.querySelector('input[type="range"][onchange*="changeTension"]');
            if (smoothnessControl) {
                smoothnessControl.parentElement.style.display = mode === 'smooth' ? 'flex' : 'none';
            }
            
            myChart.update();
        }
        
        function changeLinePattern(pattern) {
            myChart.data.datasets.forEach(dataset => {
                switch(pattern) {
                    case 'solid':
                        dataset.borderDash = [];
                        break;
                    case 'dashed':
                        dataset.borderDash = [10, 5];
                        break;
                    case 'dotted':
                        dataset.borderDash = [2, 3];
                        break;
                }
            });
            myChart.update();
        }
        
        function changeTension(value) {
            myChart.data.datasets.forEach(dataset => {
                if (!dataset.stepped) {
                    dataset.tension = parseFloat(value);
                }
            });
            myChart.update();
        }
        
        function toggleFill(isFilled) {
            if (isFilled) {
                // Fill all datasets with gradient
                myChart.data.datasets.forEach((dataset, index) => {
                    dataset.fill = 'origin';
                    // Create gradient fill
                    const ctx = myChart.ctx;
                    const gradient = ctx.createLinearGradient(0, 0, 0, myChart.height);
                    const color = dataset.borderColor;
                    gradient.addColorStop(0, color + '40');
                    gradient.addColorStop(1, color + '00');
                    dataset.backgroundColor = gradient;
                });
            } else {
                myChart.data.datasets.forEach(dataset => {
                    dataset.fill = false;
                    dataset.backgroundColor = dataset.borderColor + '20';
                });
            }
            myChart.update();
        }
        
        function togglePoints(showPoints) {
            myChart.data.datasets.forEach(dataset => {
                dataset.pointRadius = showPoints ? 3 : 0;
                dataset.pointHoverRadius = showPoints ? 5 : 0;
            });
            myChart.update();
        }
        
        function toggleBeginAtZero(beginAtZero) {
            myChart.options.scales.y.beginAtZero = beginAtZero;
            myChart.update();
        }
        
        function toggleDataset(index) {
            const meta = myChart.getDatasetMeta(index);
            meta.hidden = meta.hidden === null ? !myChart.data.datasets[index].hidden : null;
            myChart.update();
        }
        '''