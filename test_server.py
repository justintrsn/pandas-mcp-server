#!/usr/bin/env python3
"""
Fixed test script for Pandas MCP Server
Tests both the MCP layer and the core tools
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_mcp_registration():
    """Test that MCP tools are properly registered"""
    print("Testing MCP Tool Registration")
    print("=" * 50)
    
    # Import the server module
    try:
        import server
        print("✓ Server module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import server: {e}")
        return False
    
    # Test 1: Check if MCP instance exists
    print("\n1. Checking MCP instance...")
    if hasattr(server, 'mcp'):
        print("   ✓ MCP instance exists")
    else:
        print("   ✗ MCP instance not found")
        return False
    
    # Test 2: Check if tools are registered as FunctionTool objects
    print("\n2. Checking tool registration...")
    expected_tools = [
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
        'get_server_info_tool'
    ]
    
    registered_tools = []
    for tool_name in expected_tools:
        if hasattr(server, tool_name):
            tool = getattr(server, tool_name)
            # Check if it's a FunctionTool object (MCP decorated)
            if hasattr(tool, '__class__') and 'FunctionTool' in str(tool.__class__):
                print(f"   ✓ {tool_name} is registered as MCP tool")
                registered_tools.append(tool_name)
            else:
                print(f"   ✗ {tool_name} is not an MCP tool")
        else:
            print(f"   ✗ {tool_name} not found in server module")
    
    print(f"\n   Registered {len(registered_tools)}/{len(expected_tools)} tools")
    
    # Test 3: Check that core.tools functions exist
    print("\n3. Checking core.tools imports...")
    try:
        from core import tools
        print("   ✓ Core tools module imported")
        
        # Check that orchestration functions exist
        core_functions = [
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
            'get_server_info'
        ]
        
        for func_name in core_functions:
            if hasattr(tools, func_name):
                print(f"   ✓ core.tools.{func_name} exists")
            else:
                print(f"   ✗ core.tools.{func_name} not found")
    

        
    except Exception as e:
        print(f"   ✗ Failed to check core.tools: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("MCP registration checks completed!")
    
    return True


def test_core_tools():
    """Test the core tools orchestration layer directly"""
    print("\nTesting Core Tools Orchestration (Functional Tests)")
    print("=" * 50)
    
    # Import core tools
    try:
        from core import tools
        print("✓ Core tools module imported successfully")
    except Exception as e:
        print(f"✗ Failed to import core.tools: {e}")
        return False
    
    all_passed = True
    
    # Test 1: Create test data
    print("\n1. Creating test data...")
    sample_data = """product,quantity,price
Widget,10,25.50
Gadget,5,35.00
Doohickey,15,15.25"""
    
    test_file = Path("test_products.csv")
    test_file.write_text(sample_data)
    print(f"   ✓ Created {test_file}")
    
    # Create another test file for more comprehensive testing
    sample_data2 = """name,age,city,salary
Alice,30,New York,75000
Bob,25,San Francisco,85000
Charlie,35,Chicago,65000
Diana,28,Boston,70000
Eve,32,Seattle,80000"""
    
    test_file2 = Path("test_employees.csv")
    test_file2.write_text(sample_data2)
    print(f"   ✓ Created {test_file2}")
    
    # Test 2: Test read_metadata
    print("\n2. Testing core.tools.read_metadata...")
    metadata = tools.read_metadata(str(test_file))
    
    if "error" not in metadata:
        print(f"   ✓ Metadata extracted successfully")
        print(f"   ✓ Detected {metadata['structure']['columns']} columns")
        print(f"   ✓ Data quality score: {metadata['data_quality']['quality_score']:.1f}")
    else:
        print(f"   ✗ ERROR: {metadata['error']}")
        all_passed = False
    
    # Test 3: Test load_dataframe
    print("\n3. Testing core.tools.load_dataframe...")
    result = tools.load_dataframe(
        filepath=str(test_file),
        df_name="products",
        session_id="core_test"
    )
    
    if result["success"]:
        print(f"   ✓ DataFrame 'products' loaded successfully")
        print(f"   ✓ Shape: {result['dataframe_info']['shape']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Load second dataframe
    result = tools.load_dataframe(
        filepath=str(test_file2),
        df_name="employees",
        session_id="core_test"
    )
    
    if result["success"]:
        print(f"   ✓ DataFrame 'employees' loaded successfully")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 4: Test run_pandas_code
    print("\n4. Testing core.tools.run_pandas_code...")
    
    # Test calculation
    result = tools.run_pandas_code(
        code="products['total'] = products['quantity'] * products['price']",
        target_df="products",
        session_id="core_test"
    )
    
    if result["success"]:
        print(f"   ✓ Calculation executed successfully")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test aggregation
    result = tools.run_pandas_code(
        code="employees.groupby('city')['salary'].mean()",
        target_df="employees",
        session_id="core_test"
    )
    
    if result["success"]:
        print(f"   ✓ Aggregation executed successfully")
        print(f"   ✓ Result type: {result['result_type']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test describe
    result = tools.run_pandas_code(
        code="employees.describe()",
        target_df="employees",
        session_id="core_test"
    )
    
    if result["success"]:
        print(f"   ✓ Describe executed successfully")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 5: Test validate_pandas_code
    print("\n5. Testing core.tools.validate_pandas_code...")
    
    # Valid code
    result = tools.validate_pandas_code(
        code="employees.head()",
        target_df="employees",
        session_id="core_test"
    )
    
    if result.get("valid"):
        print(f"   ✓ Valid code passed validation")
    else:
        print(f"   ✗ ERROR: Valid code failed validation")
        all_passed = False
    
    # Invalid code (security violation)
    result = tools.validate_pandas_code(
        code="__import__('os').system('ls')",
        target_df="employees",
        session_id="core_test"
    )
    
    if not result.get("valid"):
        print(f"   ✓ Dangerous code correctly rejected")
    else:
        print(f"   ✗ ERROR: Dangerous code was not rejected!")
        all_passed = False
    
    # Test 6: Test list_dataframes
    print("\n6. Testing core.tools.list_dataframes...")
    result = tools.list_dataframes(session_id="core_test")
    
    if result["success"]:
        print(f"   ✓ Listed {result['total_count']} DataFrames")
        for df in result['dataframes']:
            print(f"     - {df['name']}: {df['shape']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 7: Test get_dataframe_info
    print("\n7. Testing core.tools.get_dataframe_info...")
    result = tools.get_dataframe_info("employees", "core_test")
    
    if result["success"]:
        print(f"   ✓ Got DataFrame info")
        print(f"   ✓ Memory usage: {result['dataframe_info']['memory_mb']} MB")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 8: Test get_execution_context
    print("\n8. Testing core.tools.get_execution_context...")
    result = tools.get_execution_context(session_id="core_test")
    
    if "error" not in result:
        print(f"   ✓ Execution context retrieved")
        print(f"   ✓ Available DataFrames: {result['available_dataframes']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 9: Test preview_file
    print("\n9. Testing core.tools.preview_file...")
    result = tools.preview_file(str(test_file))
    
    if result["success"]:
        print(f"   ✓ File preview successful")
        print(f"   ✓ Columns: {result['preview']['columns']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 10: Test get_supported_formats
    print("\n10. Testing core.tools.get_supported_formats...")
    result = tools.get_supported_formats()
    
    if result["success"]:
        print(f"   ✓ Got supported formats")
        print(f"   ✓ Extensions: {result['formats']['supported_extensions']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 11: Test get_session_info
    print("\n11. Testing core.tools.get_session_info...")
    result = tools.get_session_info("core_test")
    
    if result["success"]:
        print(f"   ✓ Session info retrieved")
        print(f"   ✓ DataFrames: {result['session_info']['dataframes_count']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 12: Test get_server_info
    print("\n12. Testing core.tools.get_server_info...")
    result = tools.get_server_info()
    
    if "error" not in result:
        print(f"   ✓ Server info retrieved")
        print(f"   ✓ Server: {result['server']['name']}")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Test 13: Test clear_session
    print("\n13. Testing core.tools.clear_session...")
    result = tools.clear_session(session_id="core_test")
    
    if result["success"]:
        print(f"   ✓ Session cleared")
    else:
        print(f"   ✗ ERROR: {result['error']}")
        all_passed = False
    
    # Cleanup
    test_file.unlink(missing_ok=True)
    test_file2.unlink(missing_ok=True)
    print("\n" + "=" * 50)
    print("All core tools tests completed!")
    
    return all_passed


def main():
    """Main test runner"""
    print("\n" + "=" * 60)
    print(" PANDAS MCP SERVER - COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    all_passed = True
    
    try:
        # Test MCP registration (sync)
        print("\n[PHASE 1: MCP REGISTRATION]")
        success = test_mcp_registration()
        if not success:
            all_passed = False
            print("\n❌ MCP registration tests failed!")
        else:
            print("\n✅ MCP registration tests passed!")
        
        # Test core tools (sync)
        print("\n[PHASE 2: CORE TOOLS FUNCTIONALITY]")
        success = test_core_tools()
        if not success:
            all_passed = False
            print("\n❌ Core tools tests failed!")
        else:
            print("\n✅ Core tools tests passed!")
        
        # Final summary
        print("\n" + "=" * 60)
        if all_passed:
            print("✅ ALL TESTS PASSED!")
            print("\nThe server architecture is correct:")
            print("  • server.py: Thin MCP layer (✓)")
            print("  • core.tools: Orchestration layer (✓)")
            print("  • core modules: Implementation with validation (✓)")
            print("\nYou can now run the server with:")
            print("  python server.py")
            print("\nOr test with the MCP CLI:")
            print("  npx @modelcontextprotocol/inspector python server.py")
        else:
            print("❌ SOME TESTS FAILED!")
            print("Please check the errors above.")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()