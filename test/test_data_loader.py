#!/usr/bin/env python3
"""
Test module for data loading functionality
"""

import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.data_loader import DataLoader
from core import tools
import pandas as pd
import json


def test_data_loading() -> Dict[str, Any]:
    """Test data loading functionality"""
    print("Testing Data Loading Module")
    
    results = {
        'status': 'passed',
        'total': 0,
        'passed': 0,
        'failed': 0,
        'warnings': 0,
        'details': []
    }
    
    # Initialize test data directory
    test_dir = Path(__file__).parent
    session_id = "loader_test"
    
    # Test 1: CSV loading
    print("\n1. Testing CSV loading...")
    results['total'] += 1
    
    try:
        # Create test CSV
        csv_data = """name,age,city,salary
Alice,30,New York,75000
Bob,25,San Francisco,85000
Charlie,35,Chicago,65000"""
        
        test_csv = test_dir / "test_load.csv"
        test_csv.write_text(csv_data)
        
        # Load CSV
        result = tools.load_dataframe(
            filepath=str(test_csv),
            df_name="test_csv",
            session_id=session_id
        )
        
        assert result['success'], "CSV loading failed"
        assert result['dataframe_info']['shape'] == (3, 4), "Wrong shape"
        assert 'name' in result['dataframe_info']['columns'], "Missing column"
        
        print("   ✓ CSV loading passed")
        results['passed'] += 1
        
        # Cleanup
        test_csv.unlink()
        
    except AssertionError as e:
        print(f"   ✗ CSV loading failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ CSV loading error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 2: CSV with custom delimiter
    print("\n2. Testing CSV with custom delimiter...")
    results['total'] += 1
    
    try:
        # Create TSV file
        tsv_data = """name\tage\tcity
Alice\t30\tBoston
Bob\t25\tSeattle"""
        
        test_tsv = test_dir / "test_load.tsv"
        test_tsv.write_text(tsv_data)
        
        # Load with tab delimiter
        result = tools.load_dataframe(
            filepath=str(test_tsv),
            df_name="test_tsv",
            session_id=session_id,
            delimiter='\t'
        )
        
        assert result['success'], "TSV loading failed"
        assert result['dataframe_info']['shape'] == (2, 3), "Wrong shape for TSV"
        
        print("   ✓ Custom delimiter loading passed")
        results['passed'] += 1
        
        # Cleanup
        test_tsv.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Custom delimiter loading failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Custom delimiter loading error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 3: Excel loading
    print("\n3. Testing Excel loading...")
    results['total'] += 1
    
    try:
        # Create Excel file
        test_excel = test_dir / "test_load.xlsx"
        
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [10.5, 20.5, 30.5]
        })
        
        with pd.ExcelWriter(test_excel) as writer:
            df.to_excel(writer, sheet_name='Sheet1', index=False)
            df.to_excel(writer, sheet_name='Sheet2', index=False)
        
        # Load Excel (default sheet)
        result = tools.load_dataframe(
            filepath=str(test_excel),
            df_name="test_excel",
            session_id=session_id
        )
        
        assert result['success'], "Excel loading failed"
        assert result['dataframe_info']['shape'] == (3, 3), "Wrong shape for Excel"
        
        # Load specific sheet
        result = tools.load_dataframe(
            filepath=str(test_excel),
            df_name="test_excel_sheet2",
            session_id=session_id,
            sheet_name='Sheet2'
        )
        
        assert result['success'], "Excel sheet loading failed"
        
        print("   ✓ Excel loading passed")
        results['passed'] += 1
        
        # Cleanup
        test_excel.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Excel loading failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Excel loading error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 4: JSON loading
    print("\n4. Testing JSON loading...")
    results['total'] += 1
    
    try:
        # Create JSON file
        json_data = [
            {"id": 1, "name": "Alice", "value": 100},
            {"id": 2, "name": "Bob", "value": 200},
            {"id": 3, "name": "Charlie", "value": 300}
        ]
        
        test_json = test_dir / "test_load.json"
        test_json.write_text(json.dumps(json_data))
        
        # Load JSON
        result = tools.load_dataframe(
            filepath=str(test_json),
            df_name="test_json",
            session_id=session_id
        )
        
        assert result['success'], "JSON loading failed"
        assert result['dataframe_info']['shape'] == (3, 3), "Wrong shape for JSON"
        assert 'id' in result['dataframe_info']['columns'], "Missing JSON column"
        
        print("   ✓ JSON loading passed")
        results['passed'] += 1
        
        # Cleanup
        test_json.unlink()
        
    except AssertionError as e:
        print(f"   ✗ JSON loading failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ JSON loading error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 5: Parquet loading
    print("\n5. Testing Parquet loading...")
    results['total'] += 1
    
    try:
        # Create Parquet file
        test_parquet = test_dir / "test_load.parquet"
        
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e']
        })
        
        df.to_parquet(test_parquet)
        
        # Load Parquet
        result = tools.load_dataframe(
            filepath=str(test_parquet),
            df_name="test_parquet",
            session_id=session_id
        )
        
        assert result['success'], "Parquet loading failed"
        assert result['dataframe_info']['shape'] == (5, 3), "Wrong shape for Parquet"
        
        print("   ✓ Parquet loading passed")
        results['passed'] += 1
        
        # Cleanup
        test_parquet.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Parquet loading failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Parquet loading error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 6: Preview file functionality
    print("\n6. Testing file preview...")
    results['total'] += 1
    
    try:
        # Create test file
        preview_data = """col1,col2,col3
1,A,10.5
2,B,20.5
3,C,30.5
4,D,40.5
5,E,50.5"""
        
        test_preview = test_dir / "test_preview.csv"
        test_preview.write_text(preview_data)
        
        # Preview file
        result = tools.preview_file(str(test_preview))
        
        assert result['success'], "File preview failed"
        assert 'preview' in result, "Missing preview data"
        assert result['preview']['shape'] == (5, 3), "Wrong preview shape"
        
        print("   ✓ File preview passed")
        results['passed'] += 1
        
        # Cleanup
        test_preview.unlink()
        
    except AssertionError as e:
        print(f"   ✗ File preview failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ File preview error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 7: List loaded DataFrames
    print("\n7. Testing DataFrame listing...")
    results['total'] += 1
    
    try:
        # List DataFrames
        result = tools.list_dataframes(session_id=session_id)
        
        assert result['success'], "DataFrame listing failed"
        assert result['total_count'] > 0, "No DataFrames found"
        
        # Check that our loaded DataFrames are there
        df_names = [df['name'] for df in result['dataframes']]
        expected_names = ['test_csv', 'test_tsv', 'test_excel', 'test_excel_sheet2', 'test_json', 'test_parquet']
        
        for name in expected_names:
            if name in df_names:
                print(f"     ✓ Found {name}")
        
        print("   ✓ DataFrame listing passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ DataFrame listing failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ DataFrame listing error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 8: Get DataFrame info
    print("\n8. Testing DataFrame info retrieval...")
    results['total'] += 1
    
    try:
        # Get info for a specific DataFrame
        result = tools.get_dataframe_info("test_csv", session_id)
        
        assert result['success'], "DataFrame info retrieval failed"
        assert 'dataframe_info' in result, "Missing DataFrame info"
        assert result['dataframe_info']['name'] == 'test_csv', "Wrong DataFrame name"
        
        print("   ✓ DataFrame info retrieval passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ DataFrame info retrieval failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ DataFrame info retrieval error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 9: Supported formats
    print("\n9. Testing supported formats...")
    results['total'] += 1
    
    try:
        # Get supported formats
        result = tools.get_supported_formats()
        
        assert result['success'], "Supported formats retrieval failed"
        assert 'formats' in result, "Missing formats data"
        
        extensions = result['formats']['supported_extensions']
        assert '.csv' in extensions, "CSV not in supported formats"
        assert '.xlsx' in extensions, "Excel not in supported formats"
        assert '.json' in extensions, "JSON not in supported formats"
        assert '.parquet' in extensions, "Parquet not in supported formats"
        
        print("   ✓ Supported formats passed")
        results['passed'] += 1
        
    except AssertionError as e:
        print(f"   ✗ Supported formats failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Supported formats error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Cleanup session
    try:
        tools.clear_session(session_id=session_id)
    except:
        pass
    
    # Print summary for this module
    print(f"\nData Loading Tests Summary: {results['passed']}/{results['total']} passed")
    
    return results


if __name__ == "__main__":
    # Run tests directly
    result = test_data_loading()
    
    if result['status'] == 'passed':
        print("\n✅ All data loading tests passed!")
    else:
        print(f"\n❌ {result['failed']} data loading tests failed!")
        sys.exit(1)