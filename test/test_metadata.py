#!/usr/bin/env python3
"""
Test module for metadata extraction functionality
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from core.metadata import MetadataExtractor
from core import tools


def test_metadata_extraction() -> Dict[str, Any]:
    """Test metadata extraction functionality"""
    print("Testing Metadata Extraction Module")
    
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
    
    # Test 1: Basic CSV metadata extraction
    print("\n1. Testing CSV metadata extraction...")
    results['total'] += 1
    
    try:
        # Create test CSV
        csv_data = """name,age,salary,department
John Doe,30,75000.50,Engineering
Jane Smith,25,65000.00,Marketing
Bob Johnson,35,85000.75,Sales
Alice Brown,28,70000.25,Engineering
Charlie Wilson,32,,Sales"""
        
        test_csv = test_dir / "test_metadata.csv"
        test_csv.write_text(csv_data)
        
        # Test via tools.py orchestration
        metadata = tools.read_metadata(str(test_csv))
        
        # Validate results
        assert 'file_info' in metadata, "Missing file_info"
        assert metadata['file_info']['type'] == 'csv', "Wrong file type"
        assert metadata['structure']['rows'] == 5, "Wrong row count"
        assert metadata['structure']['columns'] == 4, "Wrong column count"
        assert 'column_analysis' in metadata, "Missing column analysis"
        assert 'data_quality' in metadata, "Missing data quality"
        
        # Check data quality detection
        assert metadata['data_quality']['null_percentage'] > 0, "Should detect null values"
        
        print("   ✓ CSV metadata extraction passed")
        results['passed'] += 1
        
        # Cleanup
        test_csv.unlink()
        
    except AssertionError as e:
        print(f"   ✗ CSV metadata test failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ CSV metadata test error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 2: Excel metadata extraction
    print("\n2. Testing Excel metadata extraction...")
    results['total'] += 1
    
    try:
        # Create test Excel file using pandas
        import pandas as pd
        
        test_excel = test_dir / "test_metadata.xlsx"
        
        # Create multiple sheets
        with pd.ExcelWriter(test_excel) as writer:
            pd.DataFrame({
                'A': [1, 2, 3],
                'B': ['x', 'y', 'z']
            }).to_excel(writer, sheet_name='Sheet1', index=False)
            
            pd.DataFrame({
                'C': [4, 5, 6],
                'D': ['a', 'b', 'c']
            }).to_excel(writer, sheet_name='Sheet2', index=False)
        
        # Test metadata extraction
        relative_path = f"test/{test_excel.name}"
        metadata = tools.read_metadata(relative_path)
        
        # Validate results
        assert metadata['file_info']['type'] == 'excel', "Wrong file type"
        assert metadata['file_info']['sheet_count'] == 2, "Wrong sheet count"
        assert 'Sheet1' in metadata['sheets'], "Missing Sheet1 analysis"
        assert 'Sheet2' in metadata['sheets'], "Missing Sheet2 analysis"
        
        print("   ✓ Excel metadata extraction passed")
        results['passed'] += 1
        
        # Cleanup
        test_excel.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Excel metadata test failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Excel metadata test error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 3: JSON metadata extraction
    print("\n3. Testing JSON metadata extraction...")
    results['total'] += 1
    
    try:
        # Create test JSON
        json_data = """[
{"id": 1, "name": "Alice", "score": 95.5},
{"id": 2, "name": "Bob", "score": 87.2},
{"id": 3, "name": "Charlie", "score": 91.8}
]"""
        
        test_json = test_dir / "test_metadata.json"
        test_json.write_text(json_data)
        
        # Test metadata extraction
        relative_path = f"test/{test_json.name}"
        metadata = tools.read_metadata(relative_path)
        
        # Validate results
        assert metadata['file_info']['type'] == 'json', "Wrong file type"
        assert metadata['structure']['rows'] == 3, "Wrong row count"
        assert metadata['structure']['columns'] == 3, "Wrong column count"
        assert 'id' in metadata['column_analysis'], "Missing column analysis"
        
        print("   ✓ JSON metadata extraction passed")
        results['passed'] += 1
        
        # Cleanup
        test_json.unlink()
        
    except AssertionError as e:
        print(f"   ✗ JSON metadata test failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ JSON metadata test error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 4: Data type detection
    print("\n4. Testing data type detection...")
    results['total'] += 1
    
    try:
        # Create CSV with various data types
        mixed_data = """date,email,number,category,url
2024-01-15,john@example.com,42,Category A,https://example.com
2024-02-20,jane@test.org,37.5,Category B,http://test.org
2024-03-25,bob@demo.net,29,Category A,https://demo.net"""
        
        test_mixed = test_dir / "test_mixed.csv"
        test_mixed.write_text(mixed_data)
        
        # Test metadata extraction
        relative_path = f"test/{test_mixed.name}"
        metadata = tools.read_metadata(relative_path)
        
        # Check column type detection
        columns = metadata['column_analysis']
        
        # Date detection
        date_col = columns.get('date', {})
        assert date_col.get('primary_type') in ['datetime', 'text'], "Should detect date-like column"
        
        # Email pattern detection
        email_col = columns.get('email', {})
        if 'patterns_detected' in email_col:
            assert 'email' in email_col.get('patterns_detected', []), "Should detect email pattern"
        
        print("   ✓ Data type detection passed")
        results['passed'] += 1
        
        # Cleanup
        test_mixed.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Data type detection failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Data type detection error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 5: Large file sampling
    print("\n5. Testing large file sampling...")
    results['total'] += 1
    
    try:
        # Create a larger CSV
        import pandas as pd
        large_df = pd.DataFrame({
            'id': range(1000),
            'value': [i * 2.5 for i in range(1000)],
            'category': ['A', 'B', 'C', 'D'] * 250
        })
        
        test_large = test_dir / "test_large.csv"
        large_df.to_csv(test_large, index=False)
        
        # Test with sampling
        relative_path = f"test/{test_large.name}"
        metadata = tools.read_metadata(relative_path, sample_size=100)
        
        # Should still get full row count
        assert metadata['structure']['total_rows'] == 1000 or metadata['structure']['rows'] == 1000
        
        print("   ✓ Large file sampling passed")
        results['passed'] += 1
        
        # Cleanup
        test_large.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Large file sampling failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Large file sampling error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Test 6: Data quality analysis
    print("\n6. Testing data quality analysis...")
    results['total'] += 1
    
    try:
        # Create CSV with quality issues
        quality_data = """id,name,value
1,Alice,100
1,Alice,100
2,Bob,
3,,150
4,Charlie,200"""
        
        test_quality = test_dir / "test_quality.csv"
        test_quality.write_text(quality_data)
        
        # Test metadata extraction
        relative_path = f"test/{test_quality.name}"
        metadata = tools.read_metadata(relative_path)
        
        # Check quality metrics
        quality = metadata['data_quality']
        assert quality['duplicate_rows'] > 0, "Should detect duplicate rows"
        assert quality['null_percentage'] > 0, "Should detect null values"
        assert quality['quality_score'] < 100, "Quality score should reflect issues"
        
        print("   ✓ Data quality analysis passed")
        results['passed'] += 1
        
        # Cleanup
        test_quality.unlink()
        
    except AssertionError as e:
        print(f"   ✗ Data quality analysis failed: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    except Exception as e:
        print(f"   ✗ Data quality analysis error: {e}")
        results['failed'] += 1
        results['status'] = 'failed'
    
    # Print summary for this module
    print(f"\nMetadata Tests Summary: {results['passed']}/{results['total']} passed")
    
    return results


if __name__ == "__main__":
    # Run tests directly
    result = test_metadata_extraction()
    
    if result['status'] == 'passed':
        print("\n✅ All metadata tests passed!")
    else:
        print(f"\n❌ {result['failed']} metadata tests failed!")
        sys.exit(1)