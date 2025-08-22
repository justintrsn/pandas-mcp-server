"""
Pandas code execution module.
Handles secure execution of pandas operations on DataFrames.
"""

import logging
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np

from utils.security import CodeSecurityValidator
from utils.validators import ParameterValidator, InputValidator
from utils.formatters import format_execution_result
from storage.dataframe_manager import get_manager

logger = logging.getLogger(__name__)


class PandasExecutor:
    """
    Executes pandas code with security validation and result formatting.
    """
    
    def __init__(self):
        self.security_validator = CodeSecurityValidator()
        self.param_validator = ParameterValidator()
        self.input_validator = InputValidator()
        self.df_manager = get_manager()
    
    def execute(
        self,
        code: str,
        target_df: str = "df",
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute pandas code on a DataFrame.
        
        Args:
            code: Pandas code to execute
            target_df: Target DataFrame name
            session_id: Session identifier
            
        Returns:
            Execution result with success status and formatted output
        """
        # VALIDATION happens HERE - not in tools.py
        logger.info(f"Executing pandas code in session '{session_id}' on DataFrame '{target_df}': {code[:100]}...")
        
        # 1. Validate pandas code using ParameterValidator
        is_valid, error_msg = self.param_validator.validate_pandas_code(code)
        if not is_valid:
            logger.warning(f"Code parameter validation failed: {error_msg}")
            return {
                "success": False,
                "error": f"Code validation failed: {error_msg}"
            }
        
        # 2. Validate target DataFrame name using InputValidator
        is_valid, clean_target_df = self.input_validator.validate_dataframe_name(target_df)
        if not is_valid:
            logger.warning(f"Target DataFrame name validation failed: {clean_target_df}")
            return {
                "success": False,
                "error": f"Invalid DataFrame name: {clean_target_df}"
            }
        
        # 3. Validate session ID using InputValidator
        is_valid, clean_session_id = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            logger.warning(f"Session ID validation failed: {clean_session_id}")
            return {
                "success": False,
                "error": f"Invalid session ID: {clean_session_id}"
            }
        
        # 4. Security validation using CodeSecurityValidator
        is_safe, security_error = self.security_validator.validate_code(code)
        if not is_safe:
            logger.warning(f"Security validation failed: {security_error}")
            return {
                "success": False,
                "error": f"Security validation failed: {security_error}"
            }
        
        try:
            # Execute through DataFrame manager
            result, result_type = self.df_manager.execute_operation(
                code=code,
                target_df=clean_target_df,
                session_id=clean_session_id
            )
            
            # Format the result using utils/formatters.py
            formatted_result = format_execution_result(result, result_type)
            
            logger.info(f"Code executed successfully, result type: {result_type}")
            
            return {
                "success": True,
                "result": formatted_result,
                "result_type": result_type,
                "target_df": clean_target_df,
                "session_id": clean_session_id
            }
            
        except KeyError as e:
            error_msg = f"DataFrame '{clean_target_df}' not found in session '{clean_session_id}'"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg
            }
        except RuntimeError as e:
            # These are already formatted errors from the df_manager
            logger.error(f"Execution runtime error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "success": False,
                "error": error_msg
            }
    
    def validate_operation_only(
        self,
        code: str,
        target_df: str = "df",
        session_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Validate operation without executing it.
        
        Args:
            code: Pandas code to validate
            target_df: Target DataFrame name
            session_id: Session identifier
            
        Returns:
            Validation result
        """
        logger.info(f"Validating pandas code: {code[:100]}...")
        
        # 1. Validate pandas code
        is_valid, error_msg = self.param_validator.validate_pandas_code(code)
        if not is_valid:
            return {"valid": False, "error": f"Code validation: {error_msg}"}
        
        # 2. Validate target DataFrame name
        is_valid, clean_target_df = self.input_validator.validate_dataframe_name(target_df)
        if not is_valid:
            return {"valid": False, "error": f"DataFrame name: {clean_target_df}"}
        
        # 3. Validate session ID
        is_valid, clean_session_id = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            return {"valid": False, "error": f"Session ID: {clean_session_id}"}
        
        # 4. Security validation
        is_safe, security_error = self.security_validator.validate_code(code)
        if not is_safe:
            return {"valid": False, "error": f"Security: {security_error}"}
        
        # 5. Check if DataFrame exists
        df = self.df_manager.get_dataframe(clean_target_df, clean_session_id)
        if df is None:
            return {"valid": False, "error": f"DataFrame '{clean_target_df}' not found in session '{clean_session_id}'"}
        
        logger.info("Code validation passed")
        return {
            "valid": True,
            "message": "Code validation passed",
            "target_df": clean_target_df,
            "session_id": clean_session_id,
            "dataframe_shape": df.shape
        }
    
    def get_execution_context_info(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Get information about the execution context (available DataFrames, etc.)
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context information
        """
        # Validate session ID
        is_valid, clean_session_id = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            return {"error": f"Invalid session ID: {clean_session_id}"}
        
        try:
            # Get session info from dataframe manager
            session_info = self.df_manager.get_session_info(clean_session_id)
            
            if "error" in session_info:
                return session_info
            
            # Get list of available DataFrames
            dataframes = self.df_manager.list_dataframes(clean_session_id)
            
            # Get safe globals info
            safe_globals = self.security_validator.create_safe_globals()
            available_functions = [name for name in safe_globals.keys() if not name.startswith('_')]
            
            return {
                "session_info": session_info,
                "available_dataframes": dataframes,
                "available_functions": available_functions,
                "pandas_operations": list(self.security_validator.safe_operations)[:20],  # Sample of safe operations
                "security_restrictions": {
                    "forbidden_operations": self.security_validator.forbidden_patterns[:10],  # Sample
                    "execution_timeout": 30,
                    "memory_limit_enforced": True
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get execution context: {e}")
            return {"error": f"Failed to get execution context: {str(e)}"}


def validate_dataframe_operation_wrapper(df_name: str, operation: str) -> Dict[str, Any]:
    """
    Wrapper function for validating DataFrame operations.
    
    Args:
        df_name: Name of the DataFrame
        operation: Operation to validate
        
    Returns:
        Validation result
    """
    from utils.security import validate_dataframe_operation
    
    executor = PandasExecutor()
    
    # Validate DataFrame name
    is_valid, clean_name = executor.input_validator.validate_dataframe_name(df_name)
    if not is_valid:
        return {"valid": False, "error": f"Invalid DataFrame name: {clean_name}"}
    
    # Validate operation
    is_valid, error = validate_dataframe_operation(clean_name, operation)
    
    return {
        "valid": is_valid,
        "error": error if not is_valid else None,
        "dataframe": clean_name,
        "operation": operation
    }