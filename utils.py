"""
Shared utilities for the Holistic Horizon EPM Model project.
"""

import re
import unicodedata
import pandas as pd
import numpy as np

# --- Configuration Constants ---
TARGET_COLUMN = 'Emissions Intensity (kg CO₂ per MWh)'
TARGET_CLEANED = 'Emissions_Intensity_kg_CO2_per_MWh'
FEATURE_COLUMNS = [
    'Revenue (USD)',
    'Net Profit Margin (%)',
    'Energy Efficiency (%)',
    'Renewable Energy Share (%)',
    'Sustainability Score',
    'Innovation Index'
]
SEQUENCE_LENGTH = 10
EMBEDDING_DIMENSION = 384
BATCH_SIZE = 32
EPOCHS = 20  # Default for Phase 1, can be overridden

def clean_column_name(col_name):
    """
    Unified function to clean column names consistently across all phases.
    Handles encoding issues, special characters, and standardizes naming.
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)

    # Normalize unicode characters (e.g., remove accents, subscripts)
    col = unicodedata.normalize('NFKD', col_name)

    # Remove percent signs and parentheses first to avoid leftover characters
    col = re.sub(r'[()%]', '', col)

    # Handle known encoding issues (e.g., CO₂ -> CO2)
    col = col.replace('â\x82\x82', '2')

    # Replace any remaining non-alphanumeric (including spaces) with underscores
    col = re.sub(r'[^A-Za-z0-9%]+', '_', col)

    # Remove any leading/trailing underscores
    col = col.strip('_')

    # Remove stray percent signs and convert to a consistent lowercase name
    col = col.replace('%', 'perc')
    return col

def standardize_column_name_robust(col_name):
    """
    Alternative robust cleaning function used in later phases.
    More aggressive cleaning for consistency.
    """
    cleaned = col_name

    # 1. Handle known encoding/symbol issues (e.g., CO₂ -> CO2)
    cleaned = cleaned.replace('â\x82\x82', '2')

    # 2. Replace all non-alphanumeric, non-space characters with an underscore
    cleaned = re.sub(r'[^A-Za-z0-9\s_]', '', cleaned)

    # 3. Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')

    # 4. Collapse multiple underscores and strip leading/trailing ones
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')

    # 5. FIX: Ensure the target column is mapped correctly due to unpredictable source reading
    if 'Emissions_Intensity_kg_CO_per_MWh' in cleaned:
        return TARGET_CLEANED

    return cleaned

def load_and_clean_data(file_path, sort_by='Company_ID'):
    """
    Unified data loading and cleaning function.
    Returns cleaned DataFrame or None if failed.
    """
    print("--- Loading and Cleaning Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the file exists.")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Standardize column names using the robust function
    df.columns = [standardize_column_name_robust(col) for col in df.columns]

    # Sort by specified column if it exists
    if sort_by and sort_by in df.columns:
        df = df.sort_values(by=sort_by).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    print(f"Dataset loaded with {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def safe_str(s: str) -> str:
    """Return a console-safe ASCII-only string for Windows cmd where needed."""
    try:
        # Prefer removing non-ASCII characters to avoid encoding errors
        return re.sub(r"[^\x00-\x7F]+", "", str(s))
    except Exception:
        return str(s)
