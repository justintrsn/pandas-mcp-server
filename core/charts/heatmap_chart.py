"""
Heatmap chart implementation for Pandas MCP Server.
Visualizes correlation matrices and 2D data distributions.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.charts.base_chart import BaseChart

logger = logging.getLogger(__name__)


class HeatmapChart(BaseChart):
    """Heatmap implementation for correlation and matrix data"""
    
    def __init__(self):
        # Use matrix type for Chart.js with custom rendering
        super().__init__("matrix", "Heatmap")
    
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate data for heatmap"""
        if df.empty:
            return False, "DataFrame is empty"
        
        heatmap_type = kwargs.get('heatmap_type', 'correlation')
        
        if heatmap_type == 'correlation':
            # Need numeric columns for correlation
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) < 2:
                return False, "Correlation heatmap requires at least 2 numeric columns"
        
        elif heatmap_type == 'pivot':
            # Need columns for pivot table
            if not x_column or not y_columns:
                return False, "Pivot heatmap requires x_column and y_columns specified"
            if x_column not in df.columns:
                return False, f"Column '{x_column}' not found"
            if y_columns[0] not in df.columns:
                return False, f"Column '{y_columns[0]}' not found"
        
        elif heatmap_type == 'matrix':
            # Check if DataFrame is already matrix-like (all numeric)
            if not all(pd.api.types.is_numeric_dtype(df[col]) for col in df.columns):
                return False, "Matrix heatmap requires all numeric columns"
        
        return True, ""
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare data for heatmap"""
        heatmap_type = kwargs.get('heatmap_type', 'correlation')
        value_column = kwargs.get('value_column')
        aggregate_func = kwargs.get('aggregate_func', 'mean')
        color_scheme = kwargs.get('color_scheme', 'viridis')  # Get color scheme from kwargs
        
        if heatmap_type == 'correlation':
            # Correlation matrix
            numeric_df = df.select_dtypes(include=['number'])
            
            # Limit columns if specified
            if y_columns:
                cols = [col for col in y_columns if col in numeric_df.columns]
                numeric_df = numeric_df[cols]
            
            # Calculate correlation
            corr_matrix = numeric_df.corr()
            
            # Prepare data for Chart.js matrix
            data_points = []
            labels_x = corr_matrix.columns.tolist()
            labels_y = corr_matrix.index.tolist()
            
            for i, row_label in enumerate(labels_y):
                for j, col_label in enumerate(labels_x):
                    value = corr_matrix.iloc[i, j]
                    if not pd.isna(value):
                        data_points.append({
                            "x": col_label,
                            "y": row_label,
                            "v": round(value, 3)
                        })
            
            # Color scale for correlation (-1 to 1)
            min_value = -1
            max_value = 1
            
        elif heatmap_type == 'pivot':
            # Pivot table heatmap
            if not value_column:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    value_column = numeric_cols[0]
                else:
                    return {"error": "No numeric column for pivot values"}
            
            # Create pivot table
            pivot_table = df.pivot_table(
                index=y_columns[0] if y_columns else df.columns[0],
                columns=x_column if x_column else df.columns[1],
                values=value_column,
                aggfunc=aggregate_func
            )
            
            # Prepare data
            data_points = []
            labels_x = pivot_table.columns.astype(str).tolist()
            labels_y = pivot_table.index.astype(str).tolist()
            
            for i, row_label in enumerate(labels_y):
                for j, col_label in enumerate(labels_x):
                    value = pivot_table.iloc[i, j]
                    if not pd.isna(value):
                        data_points.append({
                            "x": col_label,
                            "y": row_label,
                            "v": round(float(value), 3)
                        })
            
            min_value = pivot_table.min().min()
            max_value = pivot_table.max().max()
            
        else:  # matrix
            # Use DataFrame as-is
            data_points = []
            labels_x = df.columns.astype(str).tolist()
            labels_y = df.index.astype(str).tolist()
            
            for i, row_label in enumerate(labels_y):
                for j, col_label in enumerate(labels_x):
                    value = df.iloc[i, j]
                    if not pd.isna(value):
                        data_points.append({
                            "x": col_label,
                            "y": row_label,
                            "v": round(float(value), 3)
                        })
            
            min_value = df.min().min()
            max_value = df.max().max()
        
        # Store color scheme info for JavaScript
        # The actual color function will be created in JavaScript
        
        return {
            "datasets": [{
                "label": "Heatmap",
                "data": data_points,
                "backgroundColor": "dynamic",  # Will be set in JavaScript
                "borderWidth": 1,
                "borderColor": "#fff",
                "width": kwargs.get('cell_width', 50),
                "height": kwargs.get('cell_height', 50)
            }],
            "labels": {
                "x": labels_x,
                "y": labels_y
            },
            "metadata": {
                "type": heatmap_type,
                "min_value": float(min_value),
                "max_value": float(max_value),
                "color_scheme": color_scheme,  # Pass color scheme name
                "shape": (len(labels_y), len(labels_x)),
                "total_cells": len(data_points)
            }
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for heatmap"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Heatmap-specific options
        options.update({
            "scales": {
                "x": {
                    "type": "category",
                    "position": "top",
                    "grid": {
                        "display": False
                    },
                    "ticks": {
                        "maxRotation": 90,
                        "minRotation": 45,
                        "autoSkip": False
                    }
                },
                "y": {
                    "type": "category",
                    "grid": {
                        "display": False
                    },
                    "ticks": {
                        "autoSkip": False
                    }
                }
            },
            "plugins": {
                **options.get("plugins", {}),
                "legend": {
                    "display": False  # Will create custom color scale
                },
                "tooltip": {
                    "enabled": True,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "callbacks": {}  # Will be set in JavaScript
                }
            },
            "aspectRatio": 1,
            "layout": {
                "padding": 10
            }
        })
        
        return options
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate heatmap specific controls"""
        base_controls = super()._generate_controls_html(chart_data)
        
        # Add heatmap-specific controls
        heatmap_controls = []
        
        # Color scheme selector
        heatmap_controls.append('''
            <div class="control-group">
                <label>Color Scheme</label>
                <select onchange="changeColorScheme(this.value)">
                    <option value="viridis">Viridis</option>
                    <option value="coolwarm">Cool-Warm</option>
                    <option value="grayscale">Grayscale</option>
                </select>
            </div>
        ''')
        
        # Cell size control
        heatmap_controls.append('''
            <div class="control-group">
                <label>Cell Size</label>
                <input type="range" min="20" max="100" step="10" value="50" 
                       onchange="changeCellSize(this.value)">
            </div>
        ''')
        
        # Show values toggle
        heatmap_controls.append('''
            <div class="control-group">
                <label>Show Values</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleValues(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Add color scale legend
        metadata = chart_data.get("metadata", {})
        min_val = metadata.get("min_value", 0)
        max_val = metadata.get("max_value", 1)
        
        color_scale_html = f'''
        <div class="color-scale" style="margin-top: 20px; padding: 10px; background: #f7f7f7; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 12px;">{min_val:.2f}</span>
                <div style="flex: 1; height: 20px; background: linear-gradient(to right, #440154, #21918c, #fde725); border-radius: 4px;"></div>
                <span style="font-size: 12px;">{max_val:.2f}</span>
            </div>
        </div>
        '''
        
        heatmap_controls.append(color_scale_html)
        
        return base_controls + '\n' + '\n'.join(heatmap_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for heatmap"""
        return '''
        // Set up color functions
        const colorSchemes = {
            viridis: function(value, min, max) {
                const normalized = (value - min) / (max - min);
                const r = Math.floor(68 + normalized * 187);
                const g = Math.floor(1 + normalized * 254);
                const b = Math.floor(84 + normalized * 171);
                return 'rgb(' + r + ', ' + g + ', ' + b + ')';
            },
            coolwarm: function(value, min, max) {
                const normalized = (value - min) / (max - min);
                if (normalized < 0.5) {
                    const intensity = normalized * 2;
                    return 'rgb(' + Math.floor(intensity * 255) + ', ' + 
                           Math.floor(intensity * 255) + ', 255)';
                } else {
                    const intensity = (1 - normalized) * 2;
                    return 'rgb(255, ' + Math.floor(intensity * 255) + ', ' + 
                           Math.floor(intensity * 255) + ')';
                }
            },
            grayscale: function(value, min, max) {
                const normalized = (value - min) / (max - min);
                const gray = Math.floor(normalized * 255);
                return 'rgb(' + gray + ', ' + gray + ', ' + gray + ')';
            }
        };
        
        // Get metadata from chart data
        const metadata = chartData.metadata || {min_value: 0, max_value: 1, color_scheme: 'viridis'};
        let currentScheme = metadata.color_scheme || 'viridis';
        
        // Set initial background color function
        myChart.data.datasets[0].backgroundColor = function(context) {
            const value = context.dataset.data[context.dataIndex].v;
            return colorSchemes[currentScheme](value, metadata.min_value, metadata.max_value);
        };
        
        // Update the chart
        myChart.update();
        
        // Custom tooltip for heatmap
        myChart.options.plugins.tooltip.callbacks.title = function(context) {
            const data = context[0].raw;
            return data.y + ' vs ' + data.x;
        };
        
        myChart.options.plugins.tooltip.callbacks.label = function(context) {
            return 'Value: ' + context.raw.v;
        };
        
        function changeColorScheme(scheme) {
            currentScheme = scheme;
            myChart.data.datasets[0].backgroundColor = function(context) {
                const value = context.dataset.data[context.dataIndex].v;
                return colorSchemes[scheme](value, metadata.min_value, metadata.max_value);
            };
            myChart.update();
        }
        
        function changeCellSize(size) {
            myChart.data.datasets[0].width = parseInt(size);
            myChart.data.datasets[0].height = parseInt(size);
            myChart.update();
        }
        
        let showValues = false;
        function toggleValues(show) {
            showValues = show;
            if (show) {
                myChart.options.plugins.datalabels = {
                    display: true,
                    color: function(context) {
                        const value = context.dataset.data[context.dataIndex].v;
                        const normalized = (value - metadata.min_value) / 
                                         (metadata.max_value - metadata.min_value);
                        return normalized > 0.5 ? '#fff' : '#000';
                    },
                    font: {
                        size: 10
                    },
                    formatter: function(value, context) {
                        return value.v.toFixed(2);
                    }
                };
            } else {
                if (myChart.options.plugins.datalabels) {
                    myChart.options.plugins.datalabels.display = false;
                }
            }
            myChart.update();
        }
        
        // Highlight row/column on hover
        let highlightedRow = null;
        let highlightedCol = null;
        
        myChart.options.onHover = function(event, elements) {
            if (elements.length > 0) {
                const data = elements[0].element.$context.raw;
                highlightedRow = data.y;
                highlightedCol = data.x;
                
                // Update opacity for highlighting
                myChart.data.datasets[0].backgroundColor = function(context) {
                    const point = context.dataset.data[context.dataIndex];
                    const baseColor = colorSchemes[currentScheme](
                        point.v, metadata.min_value, metadata.max_value
                    );
                    
                    if (point.x === highlightedCol || point.y === highlightedRow) {
                        return baseColor;
                    } else {
                        // Add transparency to non-highlighted cells
                        return baseColor.replace('rgb', 'rgba').replace(')', ', 0.3)');
                    }
                };
                myChart.update();
            } else {
                // Reset highlighting
                highlightedRow = null;
                highlightedCol = null;
                myChart.data.datasets[0].backgroundColor = function(context) {
                    const value = context.dataset.data[context.dataIndex].v;
                    return colorSchemes[currentScheme](value, metadata.min_value, metadata.max_value);
                };
                myChart.update();
            }
        };
        '''