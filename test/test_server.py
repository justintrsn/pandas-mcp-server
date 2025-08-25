#!/usr/bin/env python3
"""
Test module for MCP server registration
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def test_mcp_registration() -> Dict[str, Any]:
    """Test that MCP tools are properly registered"""
    print("Testing MCP Server Registration")
    
    results = {
        'status': 'passed',
        'total': 0,
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'details': []
    }
    
    # Test 1: Import server module
    print("\n1. Testing server module import...")
    results['total'] += 1
    
    try:
        import server
        print("   ✓ Server module imported successfully")
        results['passed'] += 1
    except Exception as e:
        print(f"   ✗ Failed to import server: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
        return results  # Can't continue without server module
    
    # Test 2: Check MCP instance
    print("\n2. Testing MCP instance...")
    results['total'] += 1
    
    if hasattr(server, 'mcp'):
        print("   ✓ MCP instance exists")
        results['passed'] += 1
    else:
        print("   ✗ MCP instance not found")
        results['failed'] += 1
        results['status'] = 'failed'
        return results  # Can't continue without MCP instance
    
    # Test 3: Check tool registration
    print("\n3. Testing tool registration...")
    
    # Include ALL tools from server.py
    expected_tools = [
        # Data tools
        'read_metadata_tool',
        'run_pandas_code_tool',
        'load_dataframe_tool',
        'list_dataframes_tool',
        'get_dataframe_info_tool',
        'validate_pandas_code_tool',
        'get_execution_context_tool',
        'preview_file_tool',
        'get_supported_formats_tool',
        'clear_session_tool',
        'get_session_info_tool',
        'get_server_info_tool',
        # Visualization tools
        'create_chart_tool',
        'suggest_charts_tool',
        'get_chart_types_tool',
        'create_correlation_heatmap_tool',
        'create_time_series_chart_tool'
    ]
    
    for tool_name in expected_tools:
        results['total'] += 1
        
        if hasattr(server, tool_name):
            tool = getattr(server, tool_name)
            # Check if it's a FunctionTool object (MCP decorated)
            if hasattr(tool, '__class__') and 'FunctionTool' in str(tool.__class__):
                print(f"   ✓ {tool_name} is registered")
                results['passed'] += 1
            else:
                print(f"   ✗ {tool_name} is not an MCP tool")
                results['failed'] += 1
                results['status'] = 'failed'
        else:
            print(f"   ✗ {tool_name} not found")
            results['failed'] += 1
            results['status'] = 'failed'
    
    # Test 4: Check core.tools imports
    print("\n4. Testing core.tools integration...")
    results['total'] += 1
    
    try:
        from core import tools
        
        # Check that core functions exist (including visualization)
        core_functions = [
            # Data functions
            'read_metadata',
            'run_pandas_code',
            'load_dataframe',
            'list_dataframes',
            'get_dataframe_info',
            'validate_pandas_code',
            'get_execution_context',
            'preview_file',
            'get_supported_formats',
            'clear_session',
            'get_session_info',
            'get_server_info',
            # Visualization functions
            'create_chart',
            'suggest_charts',
            'get_chart_types',
            'create_correlation_heatmap',
            'create_time_series_chart'
        ]
        
        missing_functions = []
        for func_name in core_functions:
            if not hasattr(tools, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"   ✗ Missing core functions: {missing_functions}")
            results['failed'] += 1
            results['status'] = 'failed'
        else:
            print(f"   ✓ All {len(core_functions)} core functions found")
            results['passed'] += 1
        
    except Exception as e:
        print(f"   ✗ Failed to check core.tools: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 5: Verify architecture
    print("\n5. Testing architecture integrity...")
    results['total'] += 1
    
    try:
        # Check that server functions are thin wrappers
        import inspect
        
        # Get source of a tool function
        source = inspect.getsource(server.read_metadata_tool)
        
        # Check if it calls tools.read_metadata (thin wrapper pattern)
        if 'tools.read_metadata' in source or 'read_metadata(' in source:
            print("   ✓ Server tools are thin wrappers (correct architecture)")
            results['passed'] += 1
        else:
            print("   ⚠ Server tools may not follow thin wrapper pattern")
            results['warnings'] += 1
            results['passed'] += 1
            
    except Exception as e:
        print(f"   ⚠ Could not verify architecture: {e}")
        results['warnings'] += 1
        results['passed'] += 1
    
    # Test 6: Check visualization tools specifically
    print("\n6. Testing visualization tools...")
    results['total'] += 1
    
    viz_tools = [
        'create_chart_tool',
        'suggest_charts_tool',
        'get_chart_types_tool',
        'create_correlation_heatmap_tool',
        'create_time_series_chart_tool'
    ]
    
    viz_functions = [
        'create_chart',
        'suggest_charts',
        'get_chart_types',
        'create_correlation_heatmap',
        'create_time_series_chart'
    ]
    
    all_viz_present = True
    for tool_name, func_name in zip(viz_tools, viz_functions):
        if not hasattr(server, tool_name):
            print(f"   ✗ Missing visualization tool: {tool_name}")
            all_viz_present = False
        if not hasattr(tools, func_name):
            print(f"   ✗ Missing visualization function: {func_name}")
            all_viz_present = False
    
    if all_viz_present:
        print("   ✓ All visualization tools are registered")
        results['passed'] += 1
    else:
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Update status if there are warnings
    if results['warnings'] > 0 and results['status'] == 'passed':
        results['status'] = 'warning'
    
    # Print summary for this module
    print(f"\nMCP Registration Tests Summary: {results['passed']}/{results['total']} passed")
    if results['warnings'] > 0:
        print(f"  ⚠ {results['warnings']} warnings")
    
    return results


if __name__ == "__main__":
    # Run tests directly
    result = test_mcp_registration()
    
    if result['status'] == 'passed':
        print("\n✅ All MCP registration tests passed!")
    elif result['status'] == 'warning':
        print(f"\n⚠️  MCP registration tests passed with {result['warnings']} warnings!")
    else:
        print(f"\n❌ {result['failed']} MCP registration tests failed!")
        sys.exit(1)