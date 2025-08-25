"""
Pie chart implementation for Pandas MCP Server.
Supports pie and doughnut charts with interactive legends.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

from core.charts.base_chart import BaseChart
from core.config import CHART_COLOR_PALETTE

logger = logging.getLogger(__name__)


class PieChart(BaseChart):
    """Pie/Doughnut chart implementation"""
    
    def __init__(self):
        super().__init__("pie", "Pie Chart")
        self.is_doughnut = False
    
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],  # Labels
        y_columns: Optional[List[str]],  # Values
        **kwargs
    ) -> Tuple[bool, str]:
        """Validate data for pie chart"""
        if df.empty:
            return False, "DataFrame is empty"
        
        # For pie charts, we need labels and at least one value column
        if y_columns and len(y_columns) > 0:
            # Check if value column exists and is numeric
            value_col = y_columns[0]
            if value_col not in df.columns:
                return False, f"Value column '{value_col}' not found"
            if not pd.api.types.is_numeric_dtype(df[value_col]):
                return False, f"Value column '{value_col}' is not numeric"
        else:
            # Need at least one numeric column
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) == 0:
                return False, "No numeric columns found for values"
        
        # Check for negative values (pie charts can't display negative values)
        if y_columns:
            value_col = y_columns[0]
            if (df[value_col] < 0).any():
                return False, "Pie charts cannot display negative values"
        
        return True, ""
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],  # Labels
        y_columns: Optional[List[str]],  # Values (uses first column)
        **kwargs
    ) -> Dict[str, Any]:
        """Prepare data for pie chart"""
        # Handle aggregation if specified
        aggregate_by = kwargs.get('aggregate_by')
        aggregate_func = kwargs.get('aggregate_func', 'sum')
        
        # Determine label column
        if x_column is None:
            # Auto-detect label column
            non_numeric = df.select_dtypes(exclude=['number']).columns
            if len(non_numeric) > 0:
                x_column = non_numeric[0]
            else:
                # Use index
                labels = [f"Item {i+1}" for i in range(len(df))]
                x_column = "Index"
        
        # Determine value column
        if y_columns is None or len(y_columns) == 0:
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if not numeric_cols:
                return {"error": "No numeric columns found for pie chart"}
            y_columns = [numeric_cols[0]]
        
        value_column = y_columns[0]
        
        # Perform aggregation if needed
        if aggregate_by and aggregate_by in df.columns:
            # Group by the specified column
            agg_df = df.groupby(aggregate_by)[value_column].agg(aggregate_func).reset_index()
            labels = agg_df[aggregate_by].astype(str).tolist()
            values = agg_df[value_column].tolist()
        elif x_column in df.columns:
            # Group by label column if there are duplicates
            if df[x_column].duplicated().any():
                agg_df = df.groupby(x_column)[value_column].sum().reset_index()
                labels = agg_df[x_column].astype(str).tolist()
                values = agg_df[value_column].tolist()
            else:
                labels = df[x_column].astype(str).tolist()
                values = df[value_column].fillna(0).tolist()
        else:
            # Labels were already created
            values = df[value_column].fillna(0).tolist()
        
        # Remove zero or negative values
        filtered_data = [(l, v) for l, v in zip(labels, values) if v > 0]
        if not filtered_data:
            return {"error": "No positive values to display in pie chart"}
        
        labels, values = zip(*filtered_data)
        
        # Limit to top N categories if too many
        max_slices = kwargs.get('max_slices', 10)
        if len(labels) > max_slices:
            # Sort by value and take top N
            sorted_pairs = sorted(zip(values, labels), reverse=True)
            top_pairs = sorted_pairs[:max_slices-1]
            other_value = sum(v for v, _ in sorted_pairs[max_slices-1:])
            
            values = [v for v, _ in top_pairs] + [other_value]
            labels = [l for _, l in top_pairs] + ["Others"]
        
        # Calculate percentages
        total = sum(values)
        percentages = [v/total * 100 for v in values]
        
        # Generate colors
        colors = [CHART_COLOR_PALETTE[i % len(CHART_COLOR_PALETTE)] for i in range(len(labels))]
        
        # Prepare dataset
        dataset = {
            "data": values,
            "backgroundColor": colors,
            "borderColor": "#fff",
            "borderWidth": 2,
            "hoverOffset": 4,
            "hoverBorderWidth": 3
        }
        
        return {
            "labels": labels,
            "datasets": [dataset],
            "metadata": {
                "value_column": value_column,
                "total": total,
                "percentages": percentages,
                "num_slices": len(labels),
                "aggregated": aggregate_by is not None
            }
        }
    
    def get_default_options(self) -> Dict[str, Any]:
        """Get default options for pie chart"""
        options = super().DEFAULT_OPTIONS.copy()
        
        # Pie-specific options
        options.update({
            "responsive": True,
            "maintainAspectRatio": True,
            "plugins": {
                **options.get("plugins", {}),
                "legend": {
                    "display": True,
                    "position": "right",
                    "labels": {
                        "padding": 15,
                        "usePointStyle": True,
                        "font": {
                            "size": 12
                        },
                        "generateLabels": None  # Will be set in JavaScript
                    }
                },
                "tooltip": {
                    "enabled": True,
                    "backgroundColor": "rgba(0, 0, 0, 0.8)",
                    "titleColor": "#fff",
                    "bodyColor": "#fff",
                    "borderColor": "#ddd",
                    "borderWidth": 1,
                    "cornerRadius": 4,
                    "padding": 10,
                    "callbacks": {}  # Will be set in JavaScript for percentage display
                },
                "datalabels": {
                    "display": False,  # Can be enabled for showing labels on slices
                    "color": "#fff",
                    "font": {
                        "weight": "bold"
                    }
                }
            },
            "cutout": "0%"  # 0% for pie, 50% for doughnut
        })
        
        return options
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate pie chart specific controls"""
        base_controls = super()._generate_controls_html(chart_data)
        
        # Add pie-specific controls
        pie_controls = []
        
        # Doughnut toggle
        pie_controls.append('''
            <div class="control-group">
                <label>Doughnut Style</label>
                <label class="switch">
                    <input type="checkbox" onchange="toggleDoughnut(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Rotation control
        pie_controls.append('''
            <div class="control-group">
                <label>Rotation</label>
                <input type="range" min="0" max="360" step="10" value="0" 
                       onchange="changeRotation(this.value)">
            </div>
        ''')
        
        # Show percentages toggle
        pie_controls.append('''
            <div class="control-group">
                <label>Show %</label>
                <label class="switch">
                    <input type="checkbox" onchange="togglePercentages(this.checked)">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Animation style
        pie_controls.append('''
            <div class="control-group">
                <label>Explode Slices</label>
                <input type="range" min="0" max="20" step="2" value="4" 
                       onchange="changeOffset(this.value)">
            </div>
        ''')
        
        return base_controls + '\n' + '\n'.join(pie_controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for pie chart"""
        return '''
        // Add percentage to tooltips
        myChart.options.plugins.tooltip.callbacks.label = function(context) {
            let label = context.label || '';
            if (label) {
                label += ': ';
            }
            const value = context.parsed;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            label += value.toLocaleString() + ' (' + percentage + '%)';
            return label;
        };
        
        // Custom legend with values
        myChart.options.plugins.legend.labels.generateLabels = function(chart) {
            const data = chart.data;
            if (data.labels.length && data.datasets.length) {
                const dataset = data.datasets[0];
                const total = dataset.data.reduce((a, b) => a + b, 0);
                
                return data.labels.map((label, i) => {
                    const value = dataset.data[i];
                    const percentage = ((value / total) * 100).toFixed(1);
                    
                    return {
                        text: label + ' (' + percentage + '%)',
                        fillStyle: dataset.backgroundColor[i],
                        strokeStyle: dataset.borderColor,
                        lineWidth: dataset.borderWidth,
                        hidden: false,
                        index: i
                    };
                });
            }
            return [];
        };
        
        myChart.update();
        
        function toggleDoughnut(isDoughnut) {
            myChart.config.type = isDoughnut ? 'doughnut' : 'pie';
            myChart.options.cutout = isDoughnut ? '50%' : '0%';
            myChart.update();
        }
        
        function changeRotation(degrees) {
            myChart.options.rotation = degrees * Math.PI / 180;
            myChart.update();
        }
        
        function togglePercentages(showPercentages) {
            if (showPercentages) {
                myChart.options.plugins.datalabels = {
                    display: true,
                    color: '#fff',
                    font: {
                        weight: 'bold',
                        size: 12
                    },
                    formatter: (value, context) => {
                        const total = context.dataset.data.reduce((a, b) => a + b, 0);
                        const percentage = ((value / total) * 100).toFixed(1);
                        return percentage + '%';
                    }
                };
            } else {
                myChart.options.plugins.datalabels.display = false;
            }
            myChart.update();
        }
        
        function changeOffset(offset) {
            myChart.data.datasets[0].hoverOffset = parseInt(offset);
            myChart.update();
        }
        
        // Sort slices
        function sortSlices(order) {
            const dataset = myChart.data.datasets[0];
            const labels = myChart.data.labels;
            const data = dataset.data;
            const backgroundColor = dataset.backgroundColor;
            
            // Create array of indices
            let indices = [];
            for (let i = 0; i < labels.length; i++) {
                indices.push({
                    label: labels[i],
                    value: data[i],
                    color: backgroundColor[i],
                    index: i
                });
            }
            
            // Sort
            if (order === 'asc') {
                indices.sort((a, b) => a.value - b.value);
            } else if (order === 'desc') {
                indices.sort((a, b) => b.value - a.value);
            } else {
                indices.sort((a, b) => a.index - b.index);
            }
            
            // Reorder
            myChart.data.labels = indices.map(item => item.label);
            dataset.data = indices.map(item => item.value);
            dataset.backgroundColor = indices.map(item => item.color);
            
            myChart.update();
        }
        
        // Explode single slice on click
        myChart.options.onClick = function(event, elements) {
            if (elements.length > 0) {
                const index = elements[0].index;
                const dataset = myChart.data.datasets[0];
                
                if (!dataset.offset) {
                    dataset.offset = new Array(dataset.data.length).fill(0);
                }
                
                // Toggle explosion
                dataset.offset[index] = dataset.offset[index] === 0 ? 20 : 0;
                myChart.update();
            }
        };
        '''