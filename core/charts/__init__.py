"""
Charts module for Pandas MCP Server.
Provides modular chart implementations for data visualization.
"""

from core.charts.base_chart import BaseChart
from core.charts.bar_chart import BarChart
from core.charts.line_chart import LineChart
from core.charts.pie_chart import PieChart
from core.charts.scatter_chart import ScatterChart
from core.charts.heatmap_chart import HeatmapChart

__all__ = [
    'BaseChart',
    'BarChart',
    'LineChart',
    'PieChart',
    'ScatterChart',
    'HeatmapChart'
]

# Chart type registry for easy lookup
CHART_TYPES = {
    'bar': BarChart,
    'line': LineChart,
    'pie': PieChart,
    'scatter': ScatterChart,
    'heatmap': HeatmapChart
}

# Supported chart types with descriptions
CHART_INFO = {
    'bar': {
        'name': 'Bar Chart',
        'description': 'Compare values across categories',
        'best_for': ['Categorical comparisons', 'Grouped data', 'Rankings']
    },
    'line': {
        'name': 'Line Chart',
        'description': 'Show trends over time or continuous data',
        'best_for': ['Time series', 'Trends', 'Multiple series comparison']
    },
    'pie': {
        'name': 'Pie/Doughnut Chart',
        'description': 'Display proportions of a whole',
        'best_for': ['Part-to-whole relationships', 'Percentages', 'Composition']
    },
    'scatter': {
        'name': 'Scatter Plot',
        'description': 'Reveal relationships between variables',
        'best_for': ['Correlations', 'Outlier detection', 'Clusters']
    },
    'heatmap': {
        'name': 'Heatmap',
        'description': 'Visualize matrix data with color intensity',
        'best_for': ['Correlation matrices', 'Pivot tables', '2D distributions']
    }
}