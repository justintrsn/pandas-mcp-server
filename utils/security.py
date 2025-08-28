"""
Security utilities for safe code execution in Pandas MCP Server
"""

import re
import ast
from typing import List, Tuple, Optional, Set
from core.config import FORBIDDEN_OPERATIONS, SAFE_PANDAS_OPERATIONS

class CodeSecurityValidator:
    """Validate and sanitize code for safe execution"""
    
    def __init__(self):
        self.forbidden_patterns = FORBIDDEN_OPERATIONS
        self.safe_operations = set(SAFE_PANDAS_OPERATIONS)
        
        # Additional patterns to check
        self.dangerous_patterns = [
            r'__[a-zA-Z]+__',  # Dunder methods
            r'lambda\s*:.*exec',  # Lambda with exec
            r'lambda\s*:.*eval',  # Lambda with eval
            r'\bimport\s+',  # Import statements
            r'from\s+.*\s+import',  # From imports
            r'\.system\s*\(',  # System calls
            r'\.popen\s*\(',  # Process opening
            r'pickle\.',  # Pickle operations
            r'marshal\.',  # Marshal operations
            r'shelve\.',  # Shelve operations
        ]
        
        # Compile regex patterns
        self.dangerous_regex = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
    
    def validate_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Validate code for security issues
        
        Args:
            code: Python code string to validate
            
        Returns:
            Tuple of (is_safe, error_message)
        """
        # Basic checks
        if not code or not code.strip():
            return False, "Empty code provided"
        
        # Check for forbidden operations
        for forbidden in self.forbidden_patterns:
            if forbidden in code.lower():
                return False, f"Forbidden operation detected: {forbidden}"
        
        # Check dangerous patterns
        for pattern in self.dangerous_regex:
            if pattern.search(code):
                return False, f"Dangerous pattern detected"
        
        # Try to parse the code as AST
        try:
            tree = ast.parse(code)
            visitor = SecurityASTVisitor(self.safe_operations)
            visitor.visit(tree)
            
            if visitor.violations:
                return False, f"Security violations found: {', '.join(visitor.violations)}"
            
        except SyntaxError as e:
            return False, f"Syntax error in code: {e}"
        except Exception as e:
            return False, f"Failed to parse code: {e}"
        
        # Check for infinite loops or excessive complexity
        if self._check_complexity(code):
            return False, "Code complexity exceeds safe limits"
        
        return True, None
    
    def _check_complexity(self, code: str) -> bool:
        """Check if code has excessive complexity"""
        # Simple heuristics for complexity
        lines = code.split('\n')
        
        # Check for excessive line count
        if len(lines) > 100:
            return True
        
        # Check for deep nesting
        max_indent = 0
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        if max_indent > 20:  # More than 5 levels of nesting (4 spaces per level)
            return True
        
        # Check for excessive loops
        loop_count = code.count('for ') + code.count('while ')
        if loop_count > 10:
            return True
        
        return False
    
    def sanitize_filepath(self, filepath: str) -> Tuple[bool, Optional[str]]:
        """
        Sanitize and validate file paths
        
        Args:
            filepath: File path to validate
            
        Returns:
            Tuple of (is_safe, sanitized_path or error_message)
        """
        # Remove any null bytes
        filepath = filepath.replace('\x00', '')
        
        # Check for path traversal attempts
        if '..' in filepath or filepath.startswith('/'):
            return False, "Path traversal detected"
        
        # Check for suspicious patterns
        suspicious_patterns = [
            r'\.\./',  # Parent directory
            r'\.\.\\',  # Parent directory (Windows)
            r'^/',  # Absolute path (Unix)
            r'^[A-Za-z]:',  # Absolute path (Windows)
            r'\\\\',  # UNC path
        ]
        
        for pattern in suspicious_patterns:
            if re.search(pattern, filepath):
                return False, "Suspicious path pattern detected"
        
        # Normalize the path
        import os
        normalized = os.path.normpath(filepath)
        
        # Ensure it doesn't escape the working directory
        if os.path.isabs(normalized):
            return False, "Absolute paths not allowed"
        
        return True, normalized
    
    def create_safe_globals(self) -> dict:
        """Create a safe globals dictionary for code execution"""
        import pandas as pd
        import numpy as np
        
        # Only include safe modules and functions
        safe_globals = {
            'pd': pd,
            'np': np,
            'DataFrame': pd.DataFrame,
            'Series': pd.Series,
            
            # Safe built-in functions
            'len': len,
            'range': range,
            'enumerate': enumerate,
            'zip': zip,
            'map': map,
            'filter': filter,
            'sum': sum,
            'min': min,
            'max': max,
            'abs': abs,
            'round': round,
            'sorted': sorted,
            'reversed': reversed,
            'all': all,
            'any': any,
            'bool': bool,
            'int': int,
            'float': float,
            'str': str,
            'list': list,
            'dict': dict,
            'tuple': tuple,
            'set': set,
            
            # Safe constants
            'True': True,
            'False': False,
            'None': None,
            
            # Restrict builtins
            '__builtins__': {
                'True': True,
                'False': False,
                'None': None,
                'len': len,
                'range': range,
                'enumerate': enumerate,
            }
        }
        
        return safe_globals


class SecurityASTVisitor(ast.NodeVisitor):
    """AST visitor to check for security violations"""
    
    def __init__(self, safe_operations: Set[str]):
        self.safe_operations = safe_operations
        self.violations = []
    
    def visit_Import(self, node):
        """Check import statements"""
        # Allow safe imports
        safe_modules = ['pandas', 'numpy', 'pd', 'np']
        for alias in node.names:
            if alias.name not in safe_modules:
                self.violations.append(f"Import of '{alias.name}' not allowed")
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        """Check from-import statements"""
        safe_modules = ['pandas', 'numpy']
        if node.module not in safe_modules:
            self.violations.append(f"Import from '{node.module}' not allowed")
        self.generic_visit(node)
    
    def visit_Call(self, node):
        """Check function calls"""
        # Check for eval/exec
        if isinstance(node.func, ast.Name):
            if node.func.id in ['eval', 'exec', 'compile', '__import__']:
                self.violations.append(f"Call to '{node.func.id}' not allowed")
        
        # Check for getattr/setattr/delattr
        if isinstance(node.func, ast.Name):
            if node.func.id in ['getattr', 'setattr', 'delattr', 'hasattr']:
                self.violations.append(f"Attribute manipulation via '{node.func.id}' not allowed")
        
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        """Check attribute access"""
        # Check for dangerous attributes
        if node.attr.startswith('_'):
            self.violations.append(f"Access to private attribute '{node.attr}' not allowed")
        
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        """Disallow async functions"""
        self.violations.append("Async functions not allowed")
        self.generic_visit(node)
    
    def visit_AsyncFor(self, node):
        """Disallow async for loops"""
        self.violations.append("Async for loops not allowed")
        self.generic_visit(node)
    
    def visit_AsyncWith(self, node):
        """Disallow async with statements"""
        self.violations.append("Async with statements not allowed")
        self.generic_visit(node)


def validate_dataframe_operation(df_name: str, operation: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a specific DataFrame operation
    
    Args:
        df_name: Name of the DataFrame
        operation: Operation to perform
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    validator = CodeSecurityValidator()
    
    # Construct the full code
    full_code = f"result = {df_name}.{operation}" if not operation.startswith(df_name) else f"result = {operation}"
    
    # Validate
    is_safe, error = validator.validate_code(full_code)
    
    return is_safe, error


def create_restricted_execution_context(dataframes: dict) -> dict:
    """
    Create a restricted execution context with loaded dataframes
    
    Args:
        dataframes: Dictionary of dataframe_name -> dataframe
        
    Returns:
        Safe globals dictionary with dataframes included
    """
    validator = CodeSecurityValidator()
    safe_globals = validator.create_safe_globals()
    
    # Add dataframes to the context
    for name, df in dataframes.items():
        if not name.startswith('_'):  # Don't allow private names
            safe_globals[name] = df
    
    return safe_globals