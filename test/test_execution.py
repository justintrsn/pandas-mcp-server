#!/usr/bin/env python3
"""
Test module for pandas code execution functionality
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.execution import PandasExecutor
from core import tools
import pandas as pd


def test_code_execution() -> Dict[str, Any]:
    """Test pandas code execution functionality"""
    print("Testing Code Execution Module")
    
    results = {
        'status': 'passed',
        'total': 0,
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'details': []
    }
    
    # Initialize test data
    test_dir = Path(__file__).parent
    session_id = "exec_test"
    
    # Setup: Create and load test DataFrames
    print("\n0. Setting up test DataFrames...")
    
    try:
        # Create test CSV
        csv_data = """product,quantity,price
Widget,10,25.50
Gadget,5,35.00
Doohickey,15,15.25"""
        
        test_csv = test_dir / "test_exec.csv"
        test_csv.write_text(csv_data)
        
        # Load DataFrame
        load_result = tools.load_dataframe(
            filepath=os.path.relpath(test_csv),
            df_name="products",
            session_id=session_id
        )
        
        assert load_result['success'], "Failed to load test DataFrame"
        print("   ✓ Test DataFrame loaded")
        
    except Exception as e:
        print(f"   ✗ Setup failed: {e}")
        return {
            'status': 'failed',
            'total': 1,
            'passed': 0,
            'failed': 1,
            'warnings': 0
        }
    
    # Test 1: Simple operations
    print("\n1. Testing simple operations...")
    results['total'] += 1
    
    try:
        # Test head()
        result = tools.run_pandas_code(
            code="products.head(2)",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "head() operation failed"
        assert result['result_type'] == 'dataframe', "Wrong result type"
        
        print("   ✓ Simple operations passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Simple operations failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 2: Aggregation operations
    print("\n2. Testing aggregation operations...")
    results['total'] += 1
    
    try:
        # Test describe()
        result = tools.run_pandas_code(
            code="products.describe()",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "describe() operation failed"
        assert result['result_type'] == 'dataframe', "Wrong result type for describe"
        
        # Test sum()
        result = tools.run_pandas_code(
            code="products['quantity'].sum()",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "sum() operation failed"
        assert result['result_type'] == 'scalar', "Wrong result type for sum"
        
        print("   ✓ Aggregation operations passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Aggregation operations failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 3: In-place modifications
    print("\n3. Testing in-place modifications...")
    results['total'] += 1
    
    try:
        # Try in-place modification (may fail)
        result = tools.run_pandas_code(
            code="products['total'] = products['quantity'] * products['price']",
            target_df="products",
            session_id=session_id
        )
        
        if not result['success']:
            print("   ⚠ In-place modification not supported (expected)")
            results['warnings'] += 1
            
            # Try alternative with assign
            result = tools.run_pandas_code(
                code="products.assign(total=products['quantity'] * products['price'])",
                target_df="products",
                session_id=session_id
            )
            
            assert result['success'], "assign() operation failed"
            print("   ✓ Alternative with assign() worked")
        else:
            print("   ✓ In-place modification passed")
        
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ In-place modifications failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 4: Filtering operations
    print("\n4. Testing filtering operations...")
    results['total'] += 1
    
    try:
        # Test query
        result = tools.run_pandas_code(
            code="products[products['quantity'] > 5]",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "Filtering operation failed"
        assert result['result_type'] == 'dataframe', "Wrong result type"
        
        print("   ✓ Filtering operations passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Filtering operations failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 5: Security validation
    print("\n5. Testing security validation...")
    results['total'] += 1
    
    try:
        # Test dangerous code rejection
        dangerous_codes = [
            "__import__('os').system('ls')",
            "exec('print(1)')",
            "eval('1+1')",
            "open('/etc/passwd')",
            "import subprocess"
        ]
        
        all_blocked = True
        for code in dangerous_codes:
            result = tools.run_pandas_code(
                code=code,
                target_df="products",
                session_id=session_id
            )
            
            if result['success']:
                print(f"   ✗ SECURITY: Dangerous code not blocked: {code}")
                all_blocked = False
                break
        
        assert all_blocked, "Some dangerous code was not blocked"
        print("   ✓ Security validation passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Security validation failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 6: Code validation without execution
    print("\n6. Testing code validation...")
    results['total'] += 1
    
    try:
        # Valid code
        result = tools.validate_pandas_code(
            code="products.head()",
            target_df="products",
            session_id=session_id
        )
        
        assert result['valid'], "Valid code marked as invalid"
        
        # Invalid code
        result = tools.validate_pandas_code(
            code="__import__('os')",
            target_df="products",
            session_id=session_id
        )
        
        assert not result['valid'], "Invalid code marked as valid"
        
        print("   ✓ Code validation passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Code validation failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 7: Complex operations
    print("\n7. Testing complex operations...")
    results['total'] += 1
    
    try:
        # GroupBy operation
        result = tools.run_pandas_code(
            code="products.groupby('product')['price'].mean()",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "GroupBy operation failed"
        assert result['result_type'] == 'series', "Wrong result type for groupby"
        
        # Multiple operations
        result = tools.run_pandas_code(
            code="products.sort_values('price').head(2)",
            target_df="products",
            session_id=session_id
        )
        
        assert result['success'], "Chained operations failed"
        
        print("   ✓ Complex operations passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Complex operations failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 8: Execution context
    print("\n8. Testing execution context...")
    results['total'] += 1
    
    try:
        # Get execution context
        context = tools.get_execution_context(session_id=session_id)
        
        assert 'available_dataframes' in context, "Missing available_dataframes"
        assert len(context['available_dataframes']) > 0, "No DataFrames in context"
        assert 'available_functions' in context, "Missing available_functions"
        
        print("   ✓ Execution context passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Execution context failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Cleanup
    try:
        tools.clear_session(session_id=session_id)
        test_csv.unlink()
    except:
        pass
    
    # Update status based on warnings
    if results['warnings'] > 0 and results['status'] == 'passed':
        results['status'] = 'warning'
    
    # Print summary for this module
    print(f"\nExecution Tests Summary: {results['passed']}/{results['total']} passed")
    if results['warnings'] > 0:
        print(f"  ⚠ {results['warnings']} warnings (in-place operations)")
    
    return results


if __name__ == "__main__":
    # Run tests directly
    result = test_code_execution()
    
    if result['status'] == 'passed':
        print("\n✅ All execution tests passed!")
    elif result['status'] == 'warning':
        print(f"\n⚠️  Execution tests passed with {result['warnings']} warnings!")
    else:
        print(f"\n❌ {result['failed']} execution tests failed!")
        sys.exit(1)