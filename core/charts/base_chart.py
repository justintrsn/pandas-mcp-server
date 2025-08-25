"""
Base chart class for all chart implementations.
Provides common functionality and HTML template generation.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

from core.config import (
    CHARTS_DIR,
    CHART_DEFAULT_WIDTH,
    CHART_DEFAULT_HEIGHT,
    CHART_COLOR_PALETTE
)

logger = logging.getLogger(__name__)


class BaseChart(ABC):
    """Abstract base class for all chart types"""
    
    # Chart.js CDN
    CHARTJS_CDN = "https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"
    
    # Default chart configuration
    DEFAULT_OPTIONS = {
        "responsive": True,
        "maintainAspectRatio": False,
        "plugins": {
            "legend": {
                "display": True,
                "position": "top"
            },
            "tooltip": {
                "enabled": True,
                "mode": "index",
                "intersect": False
            },
            "title": {
                "display": True,
                "font": {
                    "size": 16
                }
            }
        }
    }
    
    def __init__(self, chart_type: str, chart_name: str):
        """
        Initialize base chart.
        
        Args:
            chart_type: Type of chart (bar, line, pie, etc.)
            chart_name: Display name of the chart
        """
        self.chart_type = chart_type
        self.chart_name = chart_name
    
    @abstractmethod
    def validate_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Validate if the DataFrame and columns are suitable for this chart type.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def prepare_data(
        self,
        df: pd.DataFrame,
        x_column: Optional[str],
        y_columns: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare data in the format required by Chart.js.
        
        Returns:
            Dictionary with chart data or error
        """
        pass
    
    @abstractmethod
    def get_default_options(self) -> Dict[str, Any]:
        """
        Get default Chart.js options specific to this chart type.
        
        Returns:
            Dictionary of chart options
        """
        pass
    
    def generate_html(
        self,
        chart_data: Dict[str, Any],
        title: str,
        width: Optional[int] = None,
        height: Optional[int] = None,
        custom_options: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate interactive HTML with the chart and controls.
        
        Args:
            chart_data: Prepared chart data
            title: Chart title
            width: Chart width in pixels
            height: Chart height in pixels
            custom_options: Custom Chart.js options
            
        Returns:
            Complete HTML string
        """
        width = width or CHART_DEFAULT_WIDTH
        height = height or CHART_DEFAULT_HEIGHT
        
        # Merge options
        options = self.get_default_options()
        if custom_options:
            options = self._deep_merge(options, custom_options)
        
        # Add title to options
        if "plugins" not in options:
            options["plugins"] = {}
        if "title" not in options["plugins"]:
            options["plugins"]["title"] = {}
        options["plugins"]["title"]["text"] = title
        options["plugins"]["title"]["display"] = True
        
        # Generate HTML
        html_template = self._get_html_template()
        
        html_content = html_template.format(
            title=title,
            chart_type=self.chart_type,
            chartjs_cdn=self.CHARTJS_CDN,
            width=width,
            height=height,
            chart_data=json.dumps(chart_data, indent=2),
            chart_options=json.dumps(options, indent=2),
            controls_html=self._generate_controls_html(chart_data),
            custom_scripts=self._get_custom_scripts()
        )
        
        return html_content
    
    def _get_html_template(self) -> str:
        """Get the base HTML template"""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="{chartjs_cdn}"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }}
        
        .container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 30px;
            width: 100%;
            max-width: {width}px;
            animation: slideIn 0.5s ease-out;
        }}
        
        @keyframes slideIn {{
            from {{
                opacity: 0;
                transform: translateY(30px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        
        h1 {{
            color: #333;
            margin-bottom: 30px;
            text-align: center;
            font-size: 24px;
        }}
        
        .chart-container {{
            position: relative;
            height: {height}px;
            margin-bottom: 30px;
        }}
        
        .controls {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            padding: 20px;
            background: #f7f7f7;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 5px;
            min-width: 150px;
        }}
        
        .control-group label {{
            font-size: 12px;
            color: #666;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .control-group select,
        .control-group input {{
            padding: 8px 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 14px;
            transition: all 0.3s ease;
            background: white;
        }}
        
        .control-group select:hover,
        .control-group input:hover {{
            border-color: #667eea;
        }}
        
        .control-group select:focus,
        .control-group input:focus {{
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }}
        
        button {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        
        button:active {{
            transform: translateY(0);
        }}
        
        .info-panel {{
            margin-top: 20px;
            padding: 15px;
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            border-radius: 8px;
            font-size: 14px;
            color: #555;
        }}
        
        .switch {{
            position: relative;
            display: inline-block;
            width: 50px;
            height: 24px;
        }}
        
        .switch input {{
            opacity: 0;
            width: 0;
            height: 0;
        }}
        
        .slider {{
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }}
        
        .slider:before {{
            position: absolute;
            content: "";
            height: 18px;
            width: 18px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }}
        
        input:checked + .slider {{
            background-color: #667eea;
        }}
        
        input:checked + .slider:before {{
            transform: translateX(26px);
        }}
        
        @media (max-width: 768px) {{
            .container {{
                padding: 20px;
            }}
            
            h1 {{
                font-size: 20px;
            }}
            
            .controls {{
                flex-direction: column;
            }}
            
            .control-group {{
                width: 100%;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="chart-container">
            <canvas id="myChart"></canvas>
        </div>
        
        <div class="controls">
            {controls_html}
        </div>
        
        <div class="info-panel">
            <strong>Chart Type:</strong> {chart_type}<br>
            <strong>Interactive:</strong> Hover over data points for details. Use controls to customize.
        </div>
    </div>
    
    <script>
        // Chart data and options
        const chartData = {chart_data};
        const chartOptions = {chart_options};
        
        // Create chart
        const ctx = document.getElementById('myChart').getContext('2d');
        let myChart = new Chart(ctx, {{
            type: '{chart_type}',
            data: chartData,
            options: chartOptions
        }});
        
        // Update functions
        function updateChart() {{
            myChart.update();
        }}
        
        function toggleAnimation() {{
            myChart.options.animation.duration = myChart.options.animation.duration === 0 ? 1000 : 0;
            myChart.update();
        }}
        
        function changeChartType(newType) {{
            myChart.destroy();
            myChart = new Chart(ctx, {{
                type: newType,
                data: chartData,
                options: chartOptions
            }});
        }}
        
        function downloadChart() {{
            const link = document.createElement('a');
            link.download = 'chart.png';
            link.href = myChart.toBase64Image();
            link.click();
        }}
        
        function toggleLegend() {{
            myChart.options.plugins.legend.display = !myChart.options.plugins.legend.display;
            myChart.update();
        }}
        
        {custom_scripts}
    </script>
</body>
</html>'''
    
    def _generate_controls_html(self, chart_data: Dict[str, Any]) -> str:
        """Generate HTML for interactive controls"""
        controls = []
        
        # Animation toggle
        controls.append('''
            <div class="control-group">
                <label>Animation</label>
                <label class="switch">
                    <input type="checkbox" checked onchange="toggleAnimation()">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Legend toggle
        controls.append('''
            <div class="control-group">
                <label>Legend</label>
                <label class="switch">
                    <input type="checkbox" checked onchange="toggleLegend()">
                    <span class="slider"></span>
                </label>
            </div>
        ''')
        
        # Download button
        controls.append('''
            <div class="control-group">
                <label>&nbsp;</label>
                <button onclick="downloadChart()">Download</button>
            </div>
        ''')
        
        return '\n'.join(controls)
    
    def _get_custom_scripts(self) -> str:
        """Get custom JavaScript for specific chart types"""
        return ""
    
    def _deep_merge(self, dict1: dict, dict2: dict) -> dict:
        """Deep merge two dictionaries"""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_chart(
        self,
        html_content: str,
        df_name: str,
        suffix: Optional[str] = None
    ) -> Path:
        """
        Save chart HTML to file.
        
        Args:
            html_content: HTML content to save
            df_name: DataFrame name for filename
            suffix: Optional suffix for filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_str = f"_{suffix}" if suffix else ""
        filename = f"{self.chart_type}_{df_name}{suffix_str}_{timestamp}.html"
        filepath = CHARTS_DIR / filename
        
        filepath.write_text(html_content)
        logger.info(f"Chart saved to {filepath}")
        
        return filepath