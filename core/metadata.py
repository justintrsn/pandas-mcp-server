"""
File metadata extraction tool for Pandas MCP Server
Implements the read_metadata_tool functionality
"""

import pandas as pd
import numpy as np
import chardet
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import os
import logging

from core.config import (
    MAX_FILE_SIZE_MB,
    ALLOWED_FILE_EXTENSIONS,
    MAX_ROWS_PREVIEW,
    METADATA_SECTIONS,
    CHUNK_SIZE
)
from core.data_types import analyze_data_types, DataTypeDetector
from utils.validators import InputValidator, ParameterValidator
from utils.formatters import format_dataframe, format_dict

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract comprehensive metadata from data files"""
    
    def __init__(self):
        self.input_validator = InputValidator()
        self.param_validator = ParameterValidator()
        self.type_detector = DataTypeDetector()
    
    def extract_metadata(self, filepath: str, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from a file
        
        Args:
            filepath: Path to the file
            sample_size: Number of rows to sample for analysis
            
        Returns:
            Dictionary containing all metadata
        """
        # VALIDATION happens HERE - not in tools.py
        logger.info(f"Extracting metadata from: {filepath}")
        
        # Validate filepath using InputValidator
        is_valid, clean_path = self.input_validator.validate_filepath(filepath)
        if not is_valid:
            logger.error(f"Invalid filepath: {clean_path}")
            raise ValueError(f"Invalid file path: {clean_path}")
        
        # Validate sample_size using ParameterValidator
        is_valid, validated_size = self.input_validator.validate_sample_size(sample_size)
        if not is_valid:
            logger.error(f"Invalid sample size: {validated_size}")
            raise ValueError(f"Invalid sample size: {validated_size}")
        
        filepath = Path(clean_path)
        
        # Get file info
        file_size_mb = filepath.stat().st_size / 1024 / 1024
        file_extension = filepath.suffix.lower().replace('.', '')
        
        logger.info(f"File size: {file_size_mb:.2f}MB, Extension: {file_extension}")
        
        # Extract metadata based on file type
        if file_extension in ['csv', 'tsv', 'txt']:
            return self._extract_csv_metadata(filepath, file_size_mb, validated_size)
        elif file_extension in ['xlsx', 'xls']:
            return self._extract_excel_metadata(filepath, file_size_mb, validated_size)
        elif file_extension == 'json':
            return self._extract_json_metadata(filepath, file_size_mb, validated_size)
        elif file_extension == 'parquet':
            return self._extract_parquet_metadata(filepath, file_size_mb, validated_size)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_csv_metadata(self, filepath: Path, file_size_mb: float, sample_size: int) -> Dict[str, Any]:
        """Extract metadata from CSV files"""
        metadata = {
            "file_info": {
                "path": str(filepath),
                "name": filepath.name,
                "size_mb": round(file_size_mb, 2),
                "type": "csv",
                "encoding": self._detect_encoding(filepath),
            },
            "structure": {},
            "column_analysis": {},
            "data_quality": {},
            "statistics": {},
            "recommendations": [],
            "sample_data": {}
        }
        
        # Detect delimiter and other parameters
        delimiter, has_header = self._detect_csv_parameters(filepath)
        metadata["file_info"]["delimiter"] = delimiter
        metadata["file_info"]["has_header"] = has_header
        
        # Read the file
        try:
            logger.info(f"Reading CSV with delimiter '{delimiter}' and encoding '{metadata['file_info']['encoding']}'")
            
            # First, try reading with detected parameters
            df = pd.read_csv(
                filepath,
                delimiter=delimiter,
                encoding=metadata["file_info"]["encoding"],
                nrows=sample_size if file_size_mb > 10 else None,  # Sample for large files
                low_memory=False
            )
            
            # Get full row count for large files
            if file_size_mb > 10:
                total_rows = sum(1 for _ in open(filepath, 'r', encoding=metadata["file_info"]["encoding"])) - 1
                metadata["structure"]["total_rows"] = total_rows
                metadata["structure"]["sampled_rows"] = len(df)
                logger.info(f"Large file detected: sampled {len(df)} rows from {total_rows} total")
            else:
                metadata["structure"]["total_rows"] = len(df)
                metadata["structure"]["sampled_rows"] = len(df)
            
        except Exception as e:
            logger.error(f"Failed to read CSV file: {e}")
            raise RuntimeError(f"Failed to read CSV file: {e}")
        
        # Extract comprehensive metadata
        self._extract_dataframe_metadata(df, metadata)
        
        return metadata
    
    def _extract_excel_metadata(self, filepath: Path, file_size_mb: float, sample_size: int) -> Dict[str, Any]:
        """Extract metadata from Excel files"""
        metadata = {
            "file_info": {
                "path": str(filepath),
                "name": filepath.name,
                "size_mb": round(file_size_mb, 2),
                "type": "excel",
            },
            "sheets": {},
            "active_sheet_analysis": {}
        }
        
        try:
            logger.info(f"Reading Excel file: {filepath}")
            
            # Get sheet names
            excel_file = pd.ExcelFile(filepath)
            metadata["file_info"]["sheet_count"] = len(excel_file.sheet_names)
            metadata["file_info"]["sheet_names"] = excel_file.sheet_names
            
            # Analyze each sheet (or first few for large files)
            sheets_to_analyze = excel_file.sheet_names[:5]  # Limit to first 5 sheets
            
            for sheet_name in sheets_to_analyze:
                logger.info(f"Analyzing sheet: {sheet_name}")
                df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=sample_size)
                
                sheet_metadata = {
                    "structure": {
                        "rows": len(df),
                        "columns": len(df.columns)
                    },
                    "column_analysis": {},
                    "data_quality": {}
                }
                
                self._extract_dataframe_metadata(df, sheet_metadata)
                metadata["sheets"][sheet_name] = sheet_metadata
            
            # Detailed analysis of first sheet
            if sheets_to_analyze:
                first_df = pd.read_excel(filepath, sheet_name=sheets_to_analyze[0], nrows=sample_size)
                metadata["active_sheet_analysis"] = metadata["sheets"][sheets_to_analyze[0]]
                
        except Exception as e:
            logger.error(f"Failed to read Excel file: {e}")
            raise RuntimeError(f"Failed to read Excel file: {e}")
        
        return metadata
    
    def _extract_json_metadata(self, filepath: Path, file_size_mb: float, sample_size: int) -> Dict[str, Any]:
        """Extract metadata from JSON files"""
        metadata = {
            "file_info": {
                "path": str(filepath),
                "name": filepath.name,
                "size_mb": round(file_size_mb, 2),
                "type": "json",
            },
            "structure": {},
            "column_analysis": {},
            "data_quality": {}
        }
        
        try:
            logger.info(f"Reading JSON file: {filepath}")
            
            # Try different JSON orientations
            df = None
            orientations = ['records', 'columns', 'index', 'split']
            
            for orient in orientations:
                try:
                    df = pd.read_json(filepath, orient=orient, nrows=sample_size if file_size_mb > 10 else None)
                    metadata["file_info"]["json_orientation"] = orient
                    logger.info(f"Successfully parsed JSON with orientation: {orient}")
                    break
                except:
                    continue
            
            if df is None:
                # Try reading as JSON lines
                df = pd.read_json(filepath, lines=True, nrows=sample_size if file_size_mb > 10 else None)
                metadata["file_info"]["json_orientation"] = "lines"
                logger.info("Successfully parsed JSON as lines format")
            
            if df is not None:
                self._extract_dataframe_metadata(df, metadata)
            else:
                raise ValueError("Could not parse JSON file")
                
        except Exception as e:
            logger.error(f"Failed to read JSON file: {e}")
            raise RuntimeError(f"Failed to read JSON file: {e}")
        
        return metadata
    
    def _extract_parquet_metadata(self, filepath: Path, file_size_mb: float, sample_size: int) -> Dict[str, Any]:
        """Extract metadata from Parquet files"""
        metadata = {
            "file_info": {
                "path": str(filepath),
                "name": filepath.name,
                "size_mb": round(file_size_mb, 2),
                "type": "parquet",
            },
            "structure": {},
            "column_analysis": {},
            "data_quality": {},
            "parquet_metadata": {}
        }
        
        try:
            logger.info(f"Reading Parquet file: {filepath}")
            
            # Read parquet file
            df = pd.read_parquet(filepath)
            
            # Get parquet-specific metadata
            try:
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(filepath)
                
                metadata["parquet_metadata"] = {
                    "num_row_groups": parquet_file.num_row_groups,
                    "serialized_size": parquet_file.metadata.serialized_size,
                    "compression": str(parquet_file.schema_arrow)
                }
            except ImportError:
                logger.warning("PyArrow not available for detailed Parquet metadata")
                metadata["parquet_metadata"] = {"note": "PyArrow not available for detailed metadata"}
            
            # Sample if large
            if len(df) > sample_size:
                df_sample = df.sample(n=sample_size, random_state=42)
                metadata["structure"]["total_rows"] = len(df)
                metadata["structure"]["sampled_rows"] = len(df_sample)
                df = df_sample
                logger.info(f"Sampled {sample_size} rows from {metadata['structure']['total_rows']} total")
            
            self._extract_dataframe_metadata(df, metadata)
            
        except Exception as e:
            logger.error(f"Failed to read Parquet file: {e}")
            raise RuntimeError(f"Failed to read Parquet file: {e}")
        
        return metadata
    
    def _extract_dataframe_metadata(self, df: pd.DataFrame, metadata: Dict[str, Any]):
        """Extract metadata from a pandas DataFrame"""
        logger.info(f"Extracting DataFrame metadata: {df.shape}")
        
        # Basic structure
        if "structure" not in metadata:
            metadata["structure"] = {}
        
        metadata["structure"].update({
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "index_name": df.index.name,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024
        })
        
        # Column analysis using DataTypeDetector
        if METADATA_SECTIONS.get("column_analysis", True):
            metadata["column_analysis"] = self._analyze_columns(df)
        
        # Data quality analysis
        if METADATA_SECTIONS.get("data_quality", True):
            metadata["data_quality"] = self._analyze_data_quality(df)
        
        # Statistical analysis
        if METADATA_SECTIONS.get("statistics", True):
            metadata["statistics"] = self._generate_statistics(df)
        
        # Relationships
        if METADATA_SECTIONS.get("relationships", True):
            metadata["relationships"] = self._detect_relationships(df)
        
        # Recommendations
        if METADATA_SECTIONS.get("recommendations", True):
            metadata["recommendations"] = self._generate_recommendations(df, metadata)
        
        # Sample data - use formatters
        if METADATA_SECTIONS.get("sample_data", True):
            metadata["sample_data"] = self._get_sample_data(df)
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze each column using DataTypeDetector"""
        columns_metadata = {}
        
        for col in df.columns:
            col_data = df[col]
            # Use DataTypeDetector from core/data_types.py
            col_analysis = self.type_detector.detect_column_type(col_data)
            
            # Add statistical information for numeric columns
            if col_analysis["primary_type"] == "numeric":
                stats = {
                    "mean": float(col_data.mean()) if not col_data.empty else None,
                    "median": float(col_data.median()) if not col_data.empty else None,
                    "std": float(col_data.std()) if not col_data.empty else None,
                    "min": float(col_data.min()) if not col_data.empty else None,
                    "max": float(col_data.max()) if not col_data.empty else None,
                    "q25": float(col_data.quantile(0.25)) if not col_data.empty else None,
                    "q75": float(col_data.quantile(0.75)) if not col_data.empty else None,
                    "skewness": float(col_data.skew()) if len(col_data) > 1 else None,
                    "kurtosis": float(col_data.kurtosis()) if len(col_data) > 1 else None
                }
                col_analysis["statistics"] = stats
            
            # Add value counts for categorical columns
            if col_analysis["primary_type"] == "categorical" and col_analysis["unique_count"] <= 20:
                value_counts = col_data.value_counts().head(10).to_dict()
                col_analysis["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
            
            columns_metadata[col] = col_analysis
        
        return columns_metadata
    
    def _analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data quality issues"""
        total_cells = len(df) * len(df.columns)
        total_nulls = df.isnull().sum().sum()
        
        quality_analysis = {
            "total_cells": total_cells,
            "total_nulls": int(total_nulls),
            "null_percentage": float(total_nulls / total_cells * 100) if total_cells > 0 else 0,
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_row_percentage": float(df.duplicated().sum() / len(df) * 100) if len(df) > 0 else 0,
            "columns_with_nulls": df.columns[df.isnull().any()].tolist(),
            "columns_all_null": df.columns[df.isnull().all()].tolist(),
            "columns_no_null": df.columns[~df.isnull().any()].tolist(),
            "quality_score": 100.0
        }
        
        # Calculate quality score
        null_penalty = (total_nulls / total_cells) * 30 if total_cells > 0 else 0
        duplicate_penalty = (df.duplicated().sum() / len(df)) * 20 if len(df) > 0 else 0
        quality_analysis["quality_score"] = max(0, 100 - null_penalty - duplicate_penalty)
        
        # Identify specific issues
        issues = []
        if quality_analysis["duplicate_row_percentage"] > 10:
            issues.append(f"High duplicate rate: {quality_analysis['duplicate_row_percentage']:.1f}%")
        if quality_analysis["null_percentage"] > 20:
            issues.append(f"High null rate: {quality_analysis['null_percentage']:.1f}%")
        if quality_analysis["columns_all_null"]:
            issues.append(f"Columns with all nulls: {quality_analysis['columns_all_null']}")
        
        quality_analysis["issues"] = issues
        
        return quality_analysis
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summaries"""
        stats = {
            "numeric_summary": {},
            "categorical_summary": {},
            "datetime_summary": {}
        }
        
        # Numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().to_dict() if len(numeric_df.columns) > 1 else {}
            summary_stats = numeric_df.describe().to_dict()
            
            stats["numeric_summary"] = {
                "columns": numeric_df.columns.tolist(),
                "correlation_matrix": corr_matrix,
                "summary_stats": summary_stats
            }
        
        # Categorical columns
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            cat_summary = {}
            for col in categorical_df.columns:
                cat_summary[col] = {
                    "unique_values": int(categorical_df[col].nunique()),
                    "most_common": categorical_df[col].mode().iloc[0] if not categorical_df[col].mode().empty else None,
                    "cardinality": "high" if categorical_df[col].nunique() / len(df) > 0.5 else "low"
                }
            stats["categorical_summary"] = cat_summary
        
        # Datetime columns
        datetime_df = df.select_dtypes(include=['datetime64'])
        if not datetime_df.empty:
            dt_summary = {}
            for col in datetime_df.columns:
                dt_summary[col] = {
                    "min": str(datetime_df[col].min()),
                    "max": str(datetime_df[col].max()),
                    "range_days": (datetime_df[col].max() - datetime_df[col].min()).days if not datetime_df[col].empty else 0
                }
            stats["datetime_summary"] = dt_summary
        
        return stats
    
    def _detect_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect relationships between columns"""
        relationships = {
            "high_correlations": [],
            "potential_keys": [],
            "duplicate_columns": [],
            "constant_columns": []
        }
        
        # Numeric correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            corr_matrix = numeric_df.corr()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        relationships["high_correlations"].append({
                            "column1": corr_matrix.columns[i],
                            "column2": corr_matrix.columns[j],
                            "correlation": float(corr_value)
                        })
        
        # Potential keys
        for col in df.columns:
            if df[col].nunique() == len(df) and df[col].notna().all():
                relationships["potential_keys"].append(col)
        
        # Constant columns
        for col in df.columns:
            if df[col].nunique() == 1:
                relationships["constant_columns"].append(col)
        
        # Duplicate columns
        for i, col1 in enumerate(df.columns):
            for col2 in df.columns[i+1:]:
                if df[col1].equals(df[col2]):
                    relationships["duplicate_columns"].append([col1, col2])
        
        return relationships
    
    def _generate_recommendations(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Data quality recommendations
        quality = metadata.get("data_quality", {})
        if quality.get("duplicate_row_percentage", 0) > 5:
            recommendations.append(f"Remove {quality['duplicate_rows']} duplicate rows using df.drop_duplicates()")
        
        if quality.get("columns_all_null"):
            recommendations.append(f"Drop columns with all nulls: {quality['columns_all_null']}")
        
        # Type optimization recommendations
        for col, analysis in metadata.get("column_analysis", {}).items():
            if analysis.get("suggestions"):
                for suggestion in analysis["suggestions"][:2]:  # Limit suggestions per column
                    recommendations.append(f"Column '{col}': {suggestion}")
        
        # Relationship recommendations
        relationships = metadata.get("relationships", {})
        if relationships.get("high_correlations"):
            for corr in relationships["high_correlations"][:3]:  # Top 3 correlations
                recommendations.append(
                    f"High correlation ({corr['correlation']:.2f}) between '{corr['column1']}' "
                    f"and '{corr['column2']}' - consider feature selection"
                )
        
        if relationships.get("constant_columns"):
            recommendations.append(f"Drop constant columns: {relationships['constant_columns']}")
        
        # Memory optimization
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        if memory_mb > 50:
            recommendations.append("Consider using chunked processing for large dataset")
            recommendations.append("Use pd.read_csv() with 'chunksize' parameter for memory efficiency")
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def _get_sample_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get sample data for preview using formatters"""
        sample_size = min(MAX_ROWS_PREVIEW, len(df))
        
        # Use format_dataframe from utils/formatters.py for consistent formatting
        head_sample = df.head(min(5, len(df)))
        tail_sample = df.tail(min(5, len(df)))
        random_sample = df.sample(n=min(5, len(df)), random_state=42) if len(df) > 5 else pd.DataFrame()
        
        return {
            "head": head_sample.to_dict('records'),
            "tail": tail_sample.to_dict('records'),
            "random_sample": random_sample.to_dict('records') if not random_sample.empty else [],
            "preview_string": df.head(min(10, len(df))).to_string(max_cols=10, max_rows=10)
        }
    
    def _detect_encoding(self, filepath: Path) -> str:
        """Detect file encoding"""
        try:
            with open(filepath, 'rb') as f:
                result = chardet.detect(f.read(100000))
            encoding = result['encoding'] or 'utf-8'
            logger.info(f"Detected encoding: {encoding} (confidence: {result.get('confidence', 0):.2f})")
            return encoding
        except Exception as e:
            logger.warning(f"Failed to detect encoding, using utf-8: {e}")
            return 'utf-8'
    
    def _detect_csv_parameters(self, filepath: Path) -> Tuple[str, bool]:
        """Detect CSV delimiter and header presence"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                first_lines = [f.readline() for _ in range(5)]
        except Exception as e:
            logger.warning(f"Failed to read file for parameter detection: {e}")
            return ',', True
        
        # Detect delimiter
        delimiters = [',', '\t', ';', '|', ' ']
        delimiter_counts = {d: sum(line.count(d) for line in first_lines) for d in delimiters}
        delimiter = max(delimiter_counts, key=delimiter_counts.get)
        
        # Detect header (simple heuristic)
        first_line_fields = first_lines[0].strip().split(delimiter) if first_lines else []
        has_header = any(not field.replace('.', '').replace('-', '').isdigit() 
                        for field in first_line_fields)
        
        logger.info(f"Detected CSV parameters: delimiter='{delimiter}', has_header={has_header}")
        
        return delimiter, has_header


def read_metadata_tool(filepath: str) -> Dict[str, Any]:
    """
    Main tool function for reading file metadata
    
    Args:
        filepath: Path to the file to analyze
        
    Returns:
        Comprehensive metadata dictionary
    """
    extractor = MetadataExtractor()
    return extractor.extract_metadata(filepath)