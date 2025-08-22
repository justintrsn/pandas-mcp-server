#!/usr/bin/env python3
"""
Diagnostic script to identify issues with Pandas MCP Server setup
"""

import sys
import os
import platform
import subprocess
from pathlib import Path

def print_section(title):
    """Print a section header"""
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def check_python():
    """Check Python version and installation"""
    print_section("Python Environment")
    
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    print(f"Virtual Environment: {hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)}")
    
    # Check if we're in the right Python version
    if sys.version_info < (3, 10):
        print("⚠️  WARNING: Python 3.10+ is required")
        return False
    else:
        print("✓ Python version is compatible")
        return True

def check_packages():
    """Check if required packages are installed"""
    print_section("Package Installation")
    
    required_packages = {
        'fastmcp': 'FastMCP',
        'mcp': 'MCP SDK',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'uvicorn': 'Uvicorn',
        'dotenv': 'python-dotenv',
        'chardet': 'Chardet',
        'psutil': 'PSUtil'
    }
    
    missing = []
    installed = []
    
    for package, name in required_packages.items():
        try:
            if package == 'dotenv':
                __import__('dotenv')
            else:
                __import__(package)
            
            # Get version if possible
            try:
                if package == 'dotenv':
                    import dotenv
                    version = getattr(dotenv, '__version__', 'unknown')
                else:
                    mod = __import__(package)
                    version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {name:<15} {version}")
                installed.append(package)
            except:
                print(f"✓ {name:<15} (version unknown)")
                installed.append(package)
                
        except ImportError:
            print(f"✗ {name:<15} NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    else:
        print("\n✓ All required packages are installed")
        return True

def check_fastmcp_api():
    """Check FastMCP API and initialization"""
    print_section("FastMCP API Check")
    
    try:
        from fastmcp import FastMCP
        print("✓ FastMCP imported successfully")
        
        # Test different initialization methods
        print("\nTesting initialization methods:")
        
        # Method 1: Name only
        try:
            mcp1 = FastMCP("test1")
            print("✓ FastMCP('name') works")
        except Exception as e:
            print(f"✗ FastMCP('name') failed: {e}")
            return False
        
        # Method 2: With parameters (may fail)
        try:
            mcp2 = FastMCP(name="test2", version="1.0.0")
            print("✓ FastMCP(name='...', version='...') works")
        except TypeError as e:
            print(f"ℹ  FastMCP with version parameter not supported (expected)")
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
        
        # Test tool decoration
        print("\nTesting tool decoration:")
        try:
            @mcp1.tool()
            async def test_tool(param: str) -> dict:
                """Test tool"""
                return {"result": param}
            
            print("✓ Tool decoration works")
            
            # Check if tool is registered
            import asyncio
            result = asyncio.run(test_tool("test"))
            print(f"✓ Tool execution works: {result}")
            
        except Exception as e:
            print(f"✗ Tool decoration/execution failed: {e}")
            return False
        
        return True
        
    except ImportError as e:
        print(f"✗ Cannot import FastMCP: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def check_directory_structure():
    """Check if all required files and directories exist"""
    print_section("Directory Structure")
    
    required_dirs = ['core', 'utils', 'storage', 'logs', 'data']
    required_files = {
        'server.py': 'Main server',
        'core/__init__.py': 'Core package init',
        'core/config.py': 'Configuration',
        'core/data_types.py': 'Data type utilities',
        'core/metadata.py': 'Metadata extraction',
        'utils/__init__.py': 'Utils package init',
        'utils/security.py': 'Security validation',
        'storage/__init__.py': 'Storage package init',
        'storage/dataframe_manager.py': 'DataFrame manager'
    }
    
    all_good = True
    
    # Check directories
    print("Directories:")
    for dir_name in required_dirs:
        if Path(dir_name).is_dir():
            print(f"  ✓ {dir_name}/")
        else:
            print(f"  ✗ {dir_name}/ (missing)")
            all_good = False
    
    # Check files
    print("\nFiles:")
    for file_path, description in required_files.items():
        if Path(file_path).is_file():
            size = Path(file_path).stat().st_size
            print(f"  ✓ {file_path:<30} ({size} bytes)")
        else:
            print(f"  ✗ {file_path:<30} (missing)")
            all_good = False
    
    return all_good

def test_imports():
    """Test importing all modules"""
    print_section("Module Import Test")
    
    modules_to_test = [
        ('core.config', 'Configuration'),
        ('core.data_types', 'Data types'),
        ('core.metadata', 'Metadata extraction'),
        ('utils.security', 'Security'),
        ('storage.dataframe_manager', 'DataFrame manager'),
    ]
    
    all_good = True
    
    for module_name, description in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name:<25} - {description}")
        except ImportError as e:
            print(f"✗ {module_name:<25} - {e}")
            all_good = False
        except Exception as e:
            print(f"✗ {module_name:<25} - Unexpected error: {e}")
            all_good = False
    
    # Try importing the main server
    if all_good:
        print("\nTrying to import server.py...")
        try:
            import server
            print("✓ server.py imported successfully")
            
            # Check for mcp instance
            if hasattr(server, 'mcp'):
                print("✓ server.mcp instance exists")
            else:
                print("✗ server.mcp instance not found")
                all_good = False
                
        except Exception as e:
            print(f"✗ Failed to import server.py: {e}")
            all_good = False
    
    return all_good

def check_environment():
    """Check environment variables and configuration"""
    print_section("Environment Configuration")
    
    # Check for .env file
    if Path('.env').exists():
        print("✓ .env file exists")
        
        # Try to load it
        try:
            from dotenv import load_dotenv
            load_dotenv()
            print("✓ .env file loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load .env: {e}")
    else:
        print("ℹ  No .env file found (using defaults)")
    
    # Check key environment variables
    env_vars = [
        'MCP_SERVER_HOST',
        'MCP_SERVER_PORT',
        'MCP_SERVER_TRANSPORT',
        'MCP_LOG_LEVEL'
    ]
    
    print("\nEnvironment variables:")
    for var in env_vars:
        value = os.getenv(var, '(not set)')
        print(f"  {var}: {value}")
    
    return True

def suggest_fixes(results):
    """Suggest fixes based on diagnostic results"""
    print_section("Recommended Actions")
    
    if not results['python']:
        print("1. Upgrade Python to 3.10 or higher:")
        print("   brew install python@3.10  # Mac")
        print("   sudo apt install python3.10  # Ubuntu")
        print()
    
    if not results['packages']:
        print("1. Install missing packages:")
        print("   pip install -r requirements-minimal.txt")
        print("   # Or individually:")
        print("   pip install fastmcp pandas numpy uvicorn python-dotenv")
        print()
    
    if not results['directory']:
        print("2. Run the setup script to create directories:")
        print("   ./setup.sh")
        print("   # Or manually:")
        print("   mkdir -p core utils storage logs data")
        print()
    
    if not results['imports']:
        print("3. Ensure all Python files are in place")
        print("   Check that all files from the repository are copied correctly")
        print()
    
    if not results['fastmcp']:
        print("4. Update server.py to use correct FastMCP initialization:")
        print("   mcp = FastMCP('pandas-mcp-server')  # Name only")
        print()
    
    if all(results.values()):
        print("✅ Everything looks good! You should be able to run:")
        print("   python server.py")
    else:
        print("⚠️  Fix the issues above, then run this diagnostic again:")
        print("   python diagnose.py")

def main():
    """Run all diagnostics"""
    print("="*60)
    print(" PANDAS MCP SERVER - DIAGNOSTIC TOOL")
    print("="*60)
    
    results = {
        'python': check_python(),
        'packages': check_packages(),
        'directory': check_directory_structure(),
        'fastmcp': False,
        'imports': False
    }
    
    # Only check FastMCP and imports if packages are installed
    if results['packages']:
        results['fastmcp'] = check_fastmcp_api()
        
        # Only check imports if directory structure is good
        if results['directory']:
            results['imports'] = test_imports()
    
    check_environment()
    
    # Summary
    print_section("Diagnostic Summary")
    
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check.capitalize():<15} {status}")
    
    # Suggest fixes
    suggest_fixes(results)
    
    # Exit code
    if all(results.values()):
        print("\n✅ All diagnostics passed!")
        sys.exit(0)
    else:
        print("\n❌ Some diagnostics failed. See recommendations above.")
        sys.exit(1)

if __name__ == "__main__":
    main()