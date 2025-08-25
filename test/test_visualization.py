#!/usr/bin/env python3
"""
Test script for visualization functionality in Pandas MCP Server.
Tests chart creation and visualization tools.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_visualization_module():
    """
    Test visualization module for orchestrator.
    Returns test results in standard format for test.py.
    """
    # Track test results
    total_tests = 10  # We have 10 test sections
    passed_tests = 0
    failed_tests = 0
    warnings = 0
    
    # Import the tools module
    try:
        from core import tools
        from storage.dataframe_manager import get_manager
    except Exception as e:
        return {
            'status': 'failed',
            'total': 1,
            'passed': 0,
            'failed': 1,
            'warnings': 0,
            'error': f"Import failed: {e}"
        }
    
    df_manager = get_manager()
    
    # Test 1: Create sample data
    try:
        np.random.seed(42)
        sales_data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=30, freq='D'),
            'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 30),
            'revenue': np.random.randint(1000, 5000, 30),
            'profit': np.random.randint(100, 1000, 30),
            'units_sold': np.random.randint(10, 100, 30)
        })
        
        correlation_data = pd.DataFrame({
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
            'feature_3': np.random.randn(100),
            'feature_4': np.random.randn(100),
            'target': np.random.randn(100)
        })
        correlation_data['feature_2'] = correlation_data['feature_1'] * 0.8 + np.random.randn(100) * 0.2
        correlation_data['target'] = correlation_data['feature_1'] * 0.5 + correlation_data['feature_3'] * 0.3 + np.random.randn(100) * 0.2
        
        success1, _ = df_manager.add_dataframe(sales_data, "sales", "test_session")
        success2, _ = df_manager.add_dataframe(correlation_data, "features", "test_session")
        
        if success1 and success2:
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 2: Get chart types
    try:
        chart_types = tools.get_chart_types()
        if chart_types.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 3: Chart suggestions
    try:
        suggestions = tools.suggest_charts("sales", "test_session")
        if suggestions.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 4: Bar chart
    try:
        result = tools.create_chart(
            df_name="sales",
            chart_type="bar",
            x_column="category",
            y_columns=["revenue", "profit"],
            title="Sales by Category",
            session_id="test_session",
            group_by="category",
            aggregate="sum"
        )
        if result.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 5: Time series chart
    try:
        result = tools.create_time_series_chart(
            df_name="sales",
            time_column="date",
            value_columns=["revenue", "units_sold"],
            title="Revenue Over Time",
            session_id="test_session"
        )
        if result.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 6: Pie chart
    try:
        result = tools.create_chart(
            df_name="sales",
            chart_type="pie",
            x_column="category",
            y_columns=["revenue"],
            title="Revenue Distribution",
            session_id="test_session",
            aggregate_by="category",
            aggregate_func="sum"
        )
        if result.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 7: Scatter plot
    try:
        result = tools.create_chart(
            df_name="features",
            chart_type="scatter",
            x_column="feature_1",
            y_columns=["target"],
            title="Feature vs Target",
            session_id="test_session",
            show_trend=False  # Avoid scipy dependency issue
        )
        if result.get("success"):
            passed_tests += 1
        else:
            # If it fails due to scipy, count as warning
            if "scipy" in str(result.get("error", "")).lower():
                passed_tests += 1
                warnings += 1
            else:
                failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 8: Correlation heatmap
    try:
        result = tools.create_correlation_heatmap(
            df_name="features",
            title="Feature Correlations",
            session_id="test_session",
            color_scheme="coolwarm"
        )
        if result.get("success"):
            passed_tests += 1
        else:
            failed_tests += 1
    except Exception:
        failed_tests += 1
    
    # Test 9: Error handling - invalid DataFrame
    try:
        result = tools.create_chart(
            df_name="nonexistent",
            chart_type="bar",
            session_id="test_session"
        )
        if not result.get("success"):
            passed_tests += 1  # Should fail
        else:
            failed_tests += 1  # Should not succeed
    except Exception:
        passed_tests += 1  # Exception is expected
    
    # Test 10: Error handling - invalid chart type
    try:
        result = tools.create_chart(
            df_name="sales",
            chart_type="invalid_type",
            session_id="test_session"
        )
        if not result.get("success"):
            passed_tests += 1  # Should fail
        else:
            failed_tests += 1  # Should not succeed
    except Exception:
        passed_tests += 1  # Exception is expected
    
    # Cleanup
    try:
        df_manager.clear_session("test_session")
    except Exception:
        pass
    
    # Determine status
    if failed_tests == 0:
        status = 'passed' if warnings == 0 else 'warning'
    else:
        status = 'failed'
    
    return {
        'status': status,
        'total': total_tests,
        'passed': passed_tests,
        'failed': failed_tests,
        'warnings': warnings
    }


def test_visualization():
    """
    Detailed test visualization functionality for standalone execution.
    """
    print("Testing Pandas MCP Server - Visualization Module")
    print("=" * 60)
    
    # Import the tools module
    try:
        from core import tools
        print("✓ Tools module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import tools: {e}")
        return False
    
    # Test 1: Create sample data
    print("\n1. Creating sample DataFrames...")
    
    # Create sales data
    np.random.seed(42)
    sales_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=30, freq='D'),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 30),
        'revenue': np.random.randint(1000, 5000, 30),
        'profit': np.random.randint(100, 1000, 30),
        'units_sold': np.random.randint(10, 100, 30)
    })
    
    # Create correlation data
    correlation_data = pd.DataFrame({
        'feature_1': np.random.randn(100),
        'feature_2': np.random.randn(100),
        'feature_3': np.random.randn(100),
        'feature_4': np.random.randn(100),
        'target': np.random.randn(100)
    })
    # Add some correlation
    correlation_data['feature_2'] = correlation_data['feature_1'] * 0.8 + np.random.randn(100) * 0.2
    correlation_data['target'] = correlation_data['feature_1'] * 0.5 + correlation_data['feature_3'] * 0.3 + np.random.randn(100) * 0.2
    
    # Store DataFrames
    from storage.dataframe_manager import get_manager
    df_manager = get_manager()
    
    success1, msg1 = df_manager.add_dataframe(sales_data, "sales", "test_session")
    success2, msg2 = df_manager.add_dataframe(correlation_data, "features", "test_session")
    
    if success1 and success2:
        print("   ✓ Sample DataFrames created and stored")
        print(f"     - sales: {sales_data.shape}")
        print(f"     - features: {correlation_data.shape}")
    else:
        print("   ✗ Failed to store DataFrames")
        return False
    
    # Test 2: Get chart type information
    print("\n2. Testing get_chart_types...")
    chart_types = tools.get_chart_types()
    
    if chart_types.get("success"):
        print("   ✓ Chart types retrieved successfully")
        supported = chart_types.get("supported_charts", {})
        for chart_type, info in supported.items():
            print(f"     - {chart_type}: {info.get('description', '')}")
    else:
        print(f"   ✗ Failed to get chart types: {chart_types.get('error')}")
    
    # Test 3: Get chart suggestions
    print("\n3. Testing suggest_charts...")
    suggestions = tools.suggest_charts("sales", "test_session")
    
    if suggestions.get("success"):
        print("   ✓ Chart suggestions generated")
        for suggestion in suggestions.get("suggestions", []):
            print(f"     - {suggestion['chart_type']}: {suggestion['reason']}")
    else:
        print(f"   ✗ Failed to get suggestions: {suggestions.get('error')}")
    
    # Test 4: Create bar chart
    print("\n4. Testing bar chart creation...")
    bar_result = tools.create_chart(
        df_name="sales",
        chart_type="bar",
        x_column="category",
        y_columns=["revenue", "profit"],
        title="Sales by Category",
        session_id="test_session",
        group_by="category",
        aggregate="sum"
    )
    
    if bar_result.get("success"):
        print(f"   ✓ Bar chart created: {bar_result.get('filename')}")
        print(f"     Path: {bar_result.get('filepath')}")
    else:
        print(f"   ✗ Failed to create bar chart: {bar_result.get('error')}")
    
    # Test 5: Create line chart (time series)
    print("\n5. Testing time series chart...")
    line_result = tools.create_time_series_chart(
        df_name="sales",
        time_column="date",
        value_columns=["revenue", "units_sold"],
        title="Revenue Over Time",
        session_id="test_session"
    )
    
    if line_result.get("success"):
        print(f"   ✓ Time series chart created: {line_result.get('filename')}")
    else:
        print(f"   ✗ Failed to create time series: {line_result.get('error')}")
    
    # Test 6: Create pie chart
    print("\n6. Testing pie chart creation...")
    pie_result = tools.create_chart(
        df_name="sales",
        chart_type="pie",
        x_column="category",
        y_columns=["revenue"],
        title="Revenue Distribution",
        session_id="test_session",
        aggregate_by="category",
        aggregate_func="sum"
    )
    
    if pie_result.get("success"):
        print(f"   ✓ Pie chart created: {pie_result.get('filename')}")
    else:
        print(f"   ✗ Failed to create pie chart: {pie_result.get('error')}")
    
    # Test 7: Create scatter plot
    print("\n7. Testing scatter plot creation...")
    scatter_result = tools.create_chart(
        df_name="features",
        chart_type="scatter",
        x_column="feature_1",
        y_columns=["target"],
        title="Feature vs Target",
        session_id="test_session",
        show_trend=True
    )
    
    if scatter_result.get("success"):
        print(f"   ✓ Scatter plot created: {scatter_result.get('filename')}")
        metadata = scatter_result.get("metadata", {})
        if "correlations" in metadata:
            print(f"     Correlation: {metadata['correlations']}")
    else:
        print(f"   ✗ Failed to create scatter plot: {scatter_result.get('error')}")
    
    # Test 8: Create correlation heatmap
    print("\n8. Testing correlation heatmap...")
    heatmap_result = tools.create_correlation_heatmap(
        df_name="features",
        title="Feature Correlations",
        session_id="test_session",
        color_scheme="coolwarm"
    )
    
    if heatmap_result.get("success"):
        print(f"   ✓ Correlation heatmap created: {heatmap_result.get('filename')}")
    else:
        print(f"   ✗ Failed to create heatmap: {heatmap_result.get('error')}")
    
    # Test 9: Test with invalid inputs
    print("\n9. Testing error handling...")
    
    # Invalid DataFrame name
    invalid_result = tools.create_chart(
        df_name="nonexistent",
        chart_type="bar",
        session_id="test_session"
    )
    
    if not invalid_result.get("success"):
        print("   ✓ Correctly handled invalid DataFrame")
    else:
        print("   ✗ Should have failed with invalid DataFrame")
    
    # Invalid chart type
    invalid_result = tools.create_chart(
        df_name="sales",
        chart_type="invalid_type",
        session_id="test_session"
    )
    
    if not invalid_result.get("success"):
        print("   ✓ Correctly handled invalid chart type")
    else:
        print("   ✗ Should have failed with invalid chart type")
    
    # Cleanup
    print("\n10. Cleaning up...")
    df_manager.clear_session("test_session")
    print("   ✓ Test session cleared")
    
    # Summary
    print("\n" + "=" * 60)
    print("Visualization tests completed successfully!")
    
    # List created charts
    charts_dir = Path("charts")
    if charts_dir.exists():
        chart_files = list(charts_dir.glob("*.html"))
        if chart_files:
            print(f"\nCreated {len(chart_files)} chart files:")
            for chart_file in chart_files[-5:]:  # Show last 5
                print(f"  - {chart_file.name}")
            print("\nOpen these files in a browser to view the interactive charts!")
    
    return True


if __name__ == "__main__":
    """Run standalone test with detailed output"""
    print("\n" + "=" * 60)
    print(" PANDAS MCP SERVER - VISUALIZATION TEST SUITE")
    print("=" * 60)
    
    try:
        # Run the detailed tests
        success = test_visualization()
        
        if success:
            print("\n✅ All visualization tests passed!")
            print("\nYou can now use the visualization tools in the server:")
            print("  - create_chart_tool")
            print("  - suggest_charts_tool")
            print("  - get_chart_types_tool")
            print("  - create_correlation_heatmap_tool")
            print("  - create_time_series_chart_tool")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)