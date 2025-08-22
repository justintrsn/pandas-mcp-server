"""
DataFrame storage and session management for Pandas MCP Server
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from core.config import (
    MAX_DATAFRAME_SIZE_MB,
    MAX_DATAFRAMES,
    SESSION_TIMEOUT_MINUTES,
    ENABLE_CACHING,
    CACHE_TTL_SECONDS
)

class DataFrameSession:
    """Individual session for managing DataFrames"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.metadata: Dict[str, dict] = {}
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.operation_history: List[dict] = []
        self.cache: Dict[str, Any] = {}
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        timeout = timedelta(minutes=SESSION_TIMEOUT_MINUTES)
        return datetime.now() - self.last_accessed > timeout
    
    def touch(self):
        """Update last accessed time"""
        self.last_accessed = datetime.now()
    
    def add_operation(self, operation: str, df_name: str, result_type: str):
        """Add operation to history"""
        self.operation_history.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation[:200],  # Truncate long operations
            "dataframe": df_name,
            "result_type": result_type
        })
        # Keep only last 100 operations
        if len(self.operation_history) > 100:
            self.operation_history = self.operation_history[-100:]


class DataFrameManager:
    """
    Centralized DataFrame storage and management
    
    Features:
    - Session management
    - Memory limits enforcement
    - LRU eviction
    - Operation history tracking
    - Caching support
    """
    
    def __init__(self):
        self.sessions: Dict[str, DataFrameSession] = {}
        self.global_cache: Dict[str, Any] = {}
        self.total_memory_mb = 0.0
        self.max_memory_mb = MAX_DATAFRAME_SIZE_MB * MAX_DATAFRAMES
    
    def get_or_create_session(self, session_id: str = "default") -> DataFrameSession:
        """Get existing session or create new one"""
        # Clean expired sessions
        self._cleanup_expired_sessions()
        
        if session_id not in self.sessions:
            self.sessions[session_id] = DataFrameSession(session_id)
        
        session = self.sessions[session_id]
        session.touch()
        return session
    
    def add_dataframe(
        self, 
        df: pd.DataFrame, 
        name: str, 
        session_id: str = "default",
        metadata: Optional[dict] = None
    ) -> Tuple[bool, str]:
        """
        Add a DataFrame to a session
        
        Returns:
            Tuple of (success, message)
        """
        session = self.get_or_create_session(session_id)
        
        # Calculate memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Check memory limits
        if memory_mb > MAX_DATAFRAME_SIZE_MB:
            return False, f"DataFrame too large: {memory_mb:.2f}MB > {MAX_DATAFRAME_SIZE_MB}MB"
        
        # Check total memory with new dataframe
        new_total = self._calculate_total_memory() + memory_mb
        if name in session.dataframes:
            # Subtract existing dataframe size if replacing
            old_memory = session.dataframes[name].memory_usage(deep=True).sum() / 1024 / 1024
            new_total -= old_memory
        
        if new_total > self.max_memory_mb:
            # Try to free memory
            freed = self._evict_lru_dataframes(memory_mb)
            if not freed:
                return False, f"Insufficient memory: would exceed {self.max_memory_mb:.2f}MB limit"
        
        # Store dataframe
        session.dataframes[name] = df
        
        # Store metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "name": name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": memory_mb,
            "created_at": datetime.now().isoformat(),
            "hash": self._calculate_dataframe_hash(df),
            "session_id": session_id
        })
        
        session.metadata[name] = metadata
        self.total_memory_mb = self._calculate_total_memory()
        
        return True, f"DataFrame '{name}' added successfully ({memory_mb:.2f}MB)"
    
    def get_dataframe(self, name: str, session_id: str = "default") -> Optional[pd.DataFrame]:
        """Get a DataFrame by name"""
        session = self.sessions.get(session_id)
        if session and name in session.dataframes:
            session.touch()
            return session.dataframes[name]
        return None
    
    def list_dataframes(self, session_id: Optional[str] = None) -> List[dict]:
        """List all DataFrames with metadata"""
        result = []
        
        sessions_to_check = [self.sessions[session_id]] if session_id else self.sessions.values()
        
        for session in sessions_to_check:
            for name, metadata in session.metadata.items():
                result.append({
                    "name": name,
                    "session_id": session.session_id,
                    "shape": metadata["shape"],
                    "columns": len(metadata["columns"]),
                    "memory_mb": round(metadata["memory_mb"], 2),
                    "created_at": metadata["created_at"],
                    "last_accessed": session.last_accessed.isoformat()
                })
        
        return result
    
    def delete_dataframe(self, name: str, session_id: str = "default") -> bool:
        """Delete a DataFrame"""
        session = self.sessions.get(session_id)
        if session and name in session.dataframes:
            del session.dataframes[name]
            del session.metadata[name]
            self.total_memory_mb = self._calculate_total_memory()
            return True
        return False
    
    def execute_operation(
            self, 
            code: str, 
            target_df: str = "df",
            session_id: str = "default",
            timeout: int = 30
        ) -> Tuple[Any, str]:
            """
            Execute pandas operation on a DataFrame
            
            Returns:
                Tuple of (result, result_type)
            """
            import signal
            from contextlib import contextmanager
            
            @contextmanager
            def timeout_context(seconds):
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Operation timed out after {seconds} seconds")
                
                # Set the signal handler and alarm
                old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(seconds)
                try:
                    yield
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            
            session = self.get_or_create_session(session_id)
            
            # Get the target dataframe
            if target_df not in session.dataframes:
                raise KeyError(f"DataFrame '{target_df}' not found in session")
            
            # Create execution context
            from utils.security import create_restricted_execution_context
            safe_globals = create_restricted_execution_context(session.dataframes)
            
            # Execute with timeout
            try:
                with timeout_context(timeout):
                    # Store the original dataframe reference for comparison
                    original_df_id = id(session.dataframes[target_df])
                    
                    # Try to execute the code
                    exec_locals = {}
                    
                    # First, try to execute as a statement (might be assignment)
                    try:
                        exec(code, safe_globals, exec_locals)
                        
                        # Check if 'result' variable was created
                        if 'result' in exec_locals:
                            result = exec_locals['result']
                        # Check if the target dataframe was modified in place
                        elif target_df in safe_globals and id(safe_globals[target_df]) == original_df_id:
                            # DataFrame was potentially modified in place
                            # Return the modified dataframe
                            result = safe_globals[target_df]
                        else:
                            # No explicit result, try to evaluate as expression
                            try:
                                result = eval(code, safe_globals)
                            except SyntaxError:
                                # This was likely an in-place modification
                                # Return the target dataframe
                                result = session.dataframes[target_df]
                                
                    except SyntaxError:
                        # If exec fails with syntax error, try eval
                        result = eval(code, safe_globals)
                    
                    # Update the dataframe in session if it was modified
                    if target_df in safe_globals:
                        session.dataframes[target_df] = safe_globals[target_df]
                    
                    # Determine result type
                    if isinstance(result, pd.DataFrame):
                        result_type = "dataframe"
                        # Optionally store as new dataframe if it's different
                        if result is not session.dataframes[target_df]:
                            # This is a new dataframe, could store it
                            pass
                    elif isinstance(result, pd.Series):
                        result_type = "series"
                    elif isinstance(result, (list, tuple)):
                        result_type = "list"
                    elif isinstance(result, dict):
                        result_type = "dict"
                    elif isinstance(result, (int, float, str, bool)):
                        result_type = "scalar"
                    elif result is None:
                        # This might be an in-place operation
                        result_type = "none"
                        result = f"Operation completed successfully on {target_df}"
                    else:
                        result_type = "unknown"
                    
                    # Add to operation history
                    session.add_operation(code, target_df, result_type)
                    
                    return result, result_type
                    
            except TimeoutError as e:
                raise RuntimeError(str(e))
            except Exception as e:
                raise RuntimeError(f"Execution failed: {str(e)}")
    
    def get_session_info(self, session_id: str = "default") -> dict:
        """Get information about a session"""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "Session not found"}
        
        return {
            "session_id": session_id,
            "created_at": session.created_at.isoformat(),
            "last_accessed": session.last_accessed.isoformat(),
            "dataframes_count": len(session.dataframes),
            "total_memory_mb": sum(
                df.memory_usage(deep=True).sum() / 1024 / 1024 
                for df in session.dataframes.values()
            ),
            "operation_count": len(session.operation_history),
            "recent_operations": session.operation_history[-5:] if session.operation_history else []
        }
    
    def clear_session(self, session_id: str = "default") -> bool:
        """Clear all data in a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.total_memory_mb = self._calculate_total_memory()
            return True
        return False
    
    def _calculate_total_memory(self) -> float:
        """Calculate total memory usage across all sessions"""
        total = 0.0
        for session in self.sessions.values():
            for df in session.dataframes.values():
                total += df.memory_usage(deep=True).sum() / 1024 / 1024
        return total
    
    def _evict_lru_dataframes(self, required_mb: float) -> bool:
        """Evict least recently used DataFrames to free memory"""
        # Collect all dataframes with their last access time
        all_dfs = []
        for session in self.sessions.values():
            for name in session.dataframes:
                all_dfs.append((session.last_accessed, session.session_id, name))
        
        # Sort by last accessed (oldest first)
        all_dfs.sort()
        
        freed_mb = 0.0
        for last_accessed, session_id, df_name in all_dfs:
            if freed_mb >= required_mb:
                return True
            
            session = self.sessions[session_id]
            if df_name in session.dataframes:
                df_memory = session.dataframes[df_name].memory_usage(deep=True).sum() / 1024 / 1024
                del session.dataframes[df_name]
                del session.metadata[df_name]
                freed_mb += df_memory
        
        return freed_mb >= required_mb
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired = [
            sid for sid, session in self.sessions.items() 
            if session.is_expired()
        ]
        for sid in expired:
            del self.sessions[sid]
    
    def _calculate_dataframe_hash(self, df: pd.DataFrame) -> str:
        """Calculate a hash for a DataFrame for change detection"""
        # Use shape and column names for a simple hash
        hash_str = f"{df.shape}_{','.join(df.columns)}_{df.index.name}"
        return hashlib.md5(hash_str.encode()).hexdigest()[:8]
    
    def cache_result(self, key: str, value: Any, session_id: str = "default"):
        """Cache a computation result"""
        if not ENABLE_CACHING:
            return
        
        session = self.get_or_create_session(session_id)
        session.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": CACHE_TTL_SECONDS
        }
    
    def get_cached_result(self, key: str, session_id: str = "default") -> Optional[Any]:
        """Get a cached result if valid"""
        if not ENABLE_CACHING:
            return None
        
        session = self.sessions.get(session_id)
        if not session or key not in session.cache:
            return None
        
        cached = session.cache[key]
        age = (datetime.now() - cached["timestamp"]).total_seconds()
        
        if age > cached["ttl"]:
            del session.cache[key]
            return None
        
        return cached["value"]


# Global instance
_manager_instance = None

def get_manager() -> DataFrameManager:
    """Get the global DataFrameManager instance"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = DataFrameManager()
    return _manager_instance