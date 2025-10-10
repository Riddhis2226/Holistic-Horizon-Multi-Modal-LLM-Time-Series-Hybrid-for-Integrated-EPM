import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
import random

# --- Configuration (matching Phase 1) ---
FILE_PATH = r"C:\Users\acer\Downloads\Enterprise_Sustainable Power Evaluation_Dataset.csv"
EMBEDDING_DIMENSION = 384  # Standard dimension for many sentence transformers (e.g., MiniLM)

# Define the target variable (Emissions Intensity is a key EPM metric)
TARGET_COLUMN = 'Emissions Intensity (kg CO₂ per MWh)'
# Ensure this constant exactly matches the result of cleaning the TARGET_COLUMN
TARGET_CLEANED = 'Emissions_Intensity_kg_CO2_per_MWh'

# Features to use for prediction (other key EPM metrics)
FEATURE_COLUMNS = [
    'Revenue (USD)',
    'Net Profit Margin (%)',
    'Energy Efficiency (%)',
    'Renewable Energy Share (%)',
    'Sustainability Score',
    'Innovation Index'
]

def load_and_clean_data(file_path):
    """Loads and cleans the initial CSV, standardizing column names."""
    print("--- 1. Loading and Cleaning Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found. Please ensure the file is accessible.")
        return None

    # --- Robust Column Cleaning Fix ---
    def standardize_column_name(col_name):
        # 1. Start cleaning
        cleaned = col_name
        
        # 2. Aggressively clean up known problematic strings (like CO₂ encoding issues)
        cleaned = cleaned.replace('â\x82\x82', '2') # Handles one common encoding for CO₂
        
        # 3. Replace all non-alphanumeric, non-space characters with an underscore
        cleaned = re.sub(r'[^A-Za-z0-9\s_]', '', cleaned)
        
        # 4. Replace spaces with underscores
        cleaned = cleaned.replace(' ', '_')
        
        # 5. Replace multiple underscores with a single underscore
        cleaned = re.sub(r'_+', '_', cleaned)
        
        # 6. Strip any leading/trailing underscores
        cleaned = cleaned.strip('_')
        
        # FIX: Explicitly handle the target column's known problematic output 
        # to guarantee the required name, as general regex cleaning is failing.
        if 'Emissions_Intensity_kg_CO_per_MWh' in cleaned:
            return TARGET_CLEANED
        
        return cleaned

    # Apply the standardization to all columns
    df.columns = [standardize_column_name(col) for col in df.columns]

    # Convert the dataset into a single pseudo-time-series
    df_ts = df.sort_values(by='Company_ID').reset_index(drop=True)
    
    # Generate the list of *correctly* clean feature columns using the same function
    # Note: We now have to use the cleaned column names from the df.columns directly 
    # since the source string TARGET_COLUMN is unreliable.
    # We will use the columns that are necessary.
    
    # List of all clean features and targets needed
    required_cols = ['Company_ID', 'Revenue_USD', 'Net_Profit_Margin', 'Energy_Efficiency', 
                     'Renewable_Energy_Share', 'Sustainability_Score', 'Innovation_Index', 
                     TARGET_CLEANED]
    
    # Filter for the required columns
    df_ts = df_ts[required_cols].dropna()
    
    # Validation check: Ensure the critical columns exist now
    if TARGET_CLEANED not in df_ts.columns:
        print(f"\nFATAL ERROR: Target column '{TARGET_CLEANED}' is still missing after cleaning.")
        print(f"Available columns: {df_ts.columns.tolist()}")
        return None

    # Sanity Check for the other columns used later in the script
    if 'Energy_Efficiency' not in df_ts.columns or 'Innovation_Index' not in df_ts.columns:
        print("\nFATAL ERROR: Energy_Efficiency or Innovation_Index is missing after cleaning.")
        return None


    print(f"Dataset size after cleaning: {df_ts.shape}")

    return df_ts

def simulate_narratives_and_embeddings(df):
    """
    Simulates the LLM's role: generating contextual narratives and their embeddings.
    Since we cannot run a live LLM, we use rule-based simulation.
    """
    print("--- 2. Simulating Narrative Context and Embeddings ---")
    
    # 1. Calculate step-wise changes in key indicators (simulating a time-series perspective)
    # The columns here must match the names created in load_and_clean_data
    df['Emissions_Change'] = df[TARGET_CLEANED].diff().fillna(0)
    df['Efficiency_Change'] = df['Energy_Efficiency'].diff().fillna(0)
    df['Innovation_Change'] = df['Innovation_Index'].diff().fillna(0)
    
    narratives = []
    
    for i in range(len(df)):
        emissions_change = df.loc[i, 'Emissions_Change']
        efficiency_change = df.loc[i, 'Efficiency_Change']
        innovation_change = df.loc[i, 'Innovation_Change']
        
        narrative = ""

        # --- Rule-Based Narrative Generation ---
        
        # A. Significant Emissions Improvement (Negative change is good)
        if emissions_change < -50:
            narrative = "Following the deployment of a new carbon capture pilot program, the intensity of emissions saw a strong, unexpected decrease. Management expects this trend to stabilize next quarter, pending full-scale operational review."
        # B. Emissions Spike (Positive change is bad)
        elif emissions_change > 50:
            narrative = "Operational downtime at the primary renewable facility forced a temporary reliance on legacy assets, causing a sharp, but predicted, spike in emissions intensity. This is a short-term impact only."
        # C. Efficiency Drop
        elif efficiency_change < -5:
            narrative = "Initial reports indicate supply chain disruptions affecting key machinery maintenance, leading to a temporary decline in reported energy efficiency. Remedial efforts are underway."
        # D. Innovation/Future Investment
        elif innovation_change > 10:
            narrative = "Significant capital was allocated towards future-proofing and R&D for grid optimization, signalling a forward-looking strategy that may impact short-term profit margins but promises substantial long-term gains in sustainability."
        # E. Baseline/Steady State
        else:
            narrative = "Quarterly review shows stable performance across core metrics with no material changes to operational forecasts. The strategic focus remains on incremental improvements in resource allocation efficiency."

        narratives.append(narrative)

    df['Narrative'] = narratives
    
    # --- 3. Simulated Embedding Generation (Performance Fix) ---
    # Here, we simulate the embeddings using random vectors for simplicity
    print(f"Simulating {len(df)} embeddings of dimension {EMBEDDING_DIMENSION}.")
    np.random.seed(42)
    
    # Create a dummy array of embeddings (0 to 1, consistent with normalized LLM embeddings)
    embeddings = np.random.rand(len(df), EMBEDDING_DIMENSION).astype(np.float32)
    
    # FIX: Create a DataFrame from the embeddings array and concatenate it with the main DataFrame
    # This replaces the slow iterative column insertion and avoids the PerformanceWarning.
    embedding_cols = [f'Embedding_{j}' for j in range(EMBEDDING_DIMENSION)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df.index)
    
    # Join the embeddings back to the main DataFrame efficiently
    df = pd.concat([df, embeddings_df], axis=1)
        
    # Drop intermediate change columns
    df = df.drop(columns=['Emissions_Change', 'Efficiency_Change', 'Innovation_Change'])
    
    return df

def save_augmented_data(df):
    """Saves the final DataFrame containing structured data, narrative, and embeddings."""
    AUGMENTED_FILE = 'epm_augmented_data_with_embeddings.csv'
    df.to_csv(AUGMENTED_FILE, index=False)
    print(f"\n--- 3. Data Augmentation Complete ---")
    print(f"Saved augmented dataset to: {AUGMENTED_FILE}")
    print(f"Final shape: {df.shape} (Includes {EMBEDDING_DIMENSION} embedding columns)")
    print("This file is now ready for the Hybrid Model (Phase 3).")
    
if __name__ == "__main__":
    
    # The actual path to the accessible file is the string defined in FILE_PATH, 
    # not the temporary C:\Users path, which is only used for tracking the original upload.
    # We must pass the platform-accessible filename here.
    df_ts_cleaned = load_and_clean_data(FILE_PATH) 
    if df_ts_cleaned is not None:
        df_augmented = simulate_narratives_and_embeddings(df_ts_cleaned)
        save_augmented_data(df_augmented)
