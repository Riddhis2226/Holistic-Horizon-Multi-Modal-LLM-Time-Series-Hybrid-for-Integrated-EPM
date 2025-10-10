import pandas as pd
import numpy as np
import re
import random
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    # TensorFlow not available; set placeholders so the module can be imported for inspection/tests
    tf = None
    Model = None
    Input = None
    LSTM = None
    Dense = None
    Dropout = None
    Concatenate = None
    Adam = None
    TENSORFLOW_AVAILABLE = False

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# --- Configuration (MUST match Phases 1 & 2) ---
FILE_PATH = r"C:\Users\acer\OneDrive\Desktop\EPM Model\Enterprise_Sustainable Power Evaluation_Dataset.csv"
SEQUENCE_LENGTH = 10        # Historical steps for Time-Series input
EMBEDDING_DIMENSION = 384   # Dimensionality of the Narrative Embedding
BATCH_SIZE = 32
EPOCHS = 75                 # Increased epochs for better convergence of complex model

# Define the target variable
TARGET_COLUMN = 'Emissions Intensity (kg CO₂ per MWh)'
TARGET_CLEANED = 'Emissions_Intensity_kg_CO2_per_MWh'

# Features to use for prediction (Time-Series path input)
FEATURE_COLUMNS = [
    'Revenue (USD)',
    'Net Profit Margin (%)',
    'Energy Efficiency (%)',
    'Renewable Energy Share (%)',
    'Sustainability Score',
    'Innovation Index'
]

# --- Combined Data Preparation and Simulation (Replicating Phase 1 & 2 logic) ---

def generate_augmented_data(file_path):
    """
    Loads, cleans data, and simulates the LLM narratives and embeddings.
    (This function ensures the model script is self-contained and runnable).
    """
    print("--- 1. Data Loading and Narrative Simulation ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the file is in the correct directory.")
        return None, None

    # --- Robust Column Cleaning Fix (Copied from working Phase 2 logic) ---
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
    
    # List of all clean features and targets needed
    required_cols = ['Company_ID', 'Revenue_USD', 'Net_Profit_Margin', 'Energy_Efficiency', 
                     'Renewable_Energy_Share', 'Sustainability_Score', 'Innovation_Index', 
                     TARGET_CLEANED]
    
    # Filter for the required columns
    # We now use 'df_ts' directly, which has the cleaned columns from the step above.
    df_ts = df_ts[required_cols].dropna()

    # --- End of Cleaning Fix ---

    # Calculate changes for rule-based narrative generation (Phase 2 logic)
    df_ts['Emissions_Change'] = df_ts[TARGET_CLEANED].diff().fillna(0)
    df_ts['Efficiency_Change'] = df_ts['Energy_Efficiency'].diff().fillna(0)
    df_ts['Innovation_Change'] = df_ts['Innovation_Index'].diff().fillna(0)
    
    # Simulate embeddings (Phase 2 logic)
    np.random.seed(42)
    embeddings = np.random.rand(len(df_ts), EMBEDDING_DIMENSION).astype(np.float32)
    
    # FIX: Use efficient concatenation to avoid PerformanceWarning
    embedding_cols = [f'Embedding_{j}' for j in range(EMBEDDING_DIMENSION)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df_ts.index)
    df_ts = pd.concat([df_ts, embeddings_df], axis=1)

    df_ts = df_ts.drop(columns=['Emissions_Change', 'Efficiency_Change', 'Innovation_Change', 'Company_ID'])
    
    # Identify feature and embedding columns
    ts_features = [col for col in df_ts.columns if col.startswith(('Revenue', 'Net_Profit', 'Energy_Efficiency', 'Renewable', 'Sustainability_Score', 'Innovation_Index'))]
    # The embedding features start after the TS features
    num_ts_features = len(ts_features)

    # Final data scaling and structuring
    data = df_ts.values
    
    # 1. Scale Features
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(data[:, :-1]) # All except target
    
    # 2. Scale Target
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(data[:, -1].reshape(-1, 1))

    # Recombine scaled data
    scaled_data = np.hstack((features_scaled, target_scaled))
    
    return scaled_data, num_ts_features, scaler_target


def create_hybrid_sequences(data, sequence_length, num_ts_features):
    """
    Creates sequences for the dual-input model.
    X_ts: Time-series features (multi-step history)
    X_narrative: Narrative embedding (single step at prediction time)
    y: Target value (single step at prediction time)
    """
    X_ts, X_narrative, y = [], [], []
    
    # Number of total non-target columns (TS Features + Embeddings)
    num_total_features = data.shape[1] - 1 
    
    for i in range(len(data) - sequence_length):
        # A. Time-Series Input (Historical sequence of TS Features only)
        # Select rows [i to i + SEQUENCE_LENGTH - 1] and columns [0 to num_ts_features - 1]
        X_ts.append(data[i: i + sequence_length, :num_ts_features])
        
        # B. Narrative Input (Embedding vector corresponding to the prediction step)
        # Select row [i + SEQUENCE_LENGTH] and columns [num_ts_features to num_total_features - 1]
        X_narrative.append(data[i + sequence_length, num_ts_features:num_total_features])
        
        # C. Target (Target value at the prediction step)
        # Select row [i + SEQUENCE_LENGTH] and the last column
        y.append(data[i + sequence_length, -1])

    return np.array(X_ts), np.array(X_narrative), np.array(y)

# --- 2. Hybrid Fusion Model Definition ---

def build_hybrid_model(ts_input_shape, narrative_input_shape):
    """
    Defines the Dual-Input Fusion Model architecture using Keras Functional API.
    """
    print("\n--- 2. Building Dual-Input Hybrid Fusion Model ---")
    
    # 1. Time-Series Path (Quantitative)
    ts_input = Input(shape=ts_input_shape, name='time_series_input')
    lstm_1 = LSTM(units=64, return_sequences=True)(ts_input)
    dropout_ts_1 = Dropout(0.3)(lstm_1)
    lstm_2 = LSTM(units=32)(dropout_ts_1)
    ts_output = Dense(16, activation='relu', name='ts_feature_vector')(lstm_2)
    
    # 2. Narrative Path (Qualitative)
    narrative_input = Input(shape=narrative_input_shape, name='narrative_embedding_input')
    dense_narrative_1 = Dense(64, activation='relu')(narrative_input)
    dropout_narrative_1 = Dropout(0.3)(dense_narrative_1)
    narrative_output = Dense(16, activation='relu', name='narrative_feature_vector')(dropout_narrative_1)
    
    # 3. Fusion Layer (Concatenation)
    fusion_layer = Concatenate(name='fusion_layer')([ts_output, narrative_output])
    
    # 4. Final Prediction Head
    dense_final_1 = Dense(16, activation='relu')(fusion_layer)
    output = Dense(1, activation='linear', name='emissions_prediction')(dense_final_1)
    
    # Define the final model
    model = Model(inputs=[ts_input, narrative_input], outputs=output, name='Holistic_Horizon_EPM_Hybrid')
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Prepare and structure data
    scaled_data, num_ts_features, scaler_target = generate_augmented_data(FILE_PATH)
    
    if scaled_data is None:
        exit()

    X_ts, X_narrative, y = create_hybrid_sequences(scaled_data, SEQUENCE_LENGTH, num_ts_features)

    # 2. Split into train and test sets
    test_size = 0.2
    split_index = int(len(X_ts) * (1 - test_size))

    X_ts_train, X_ts_test = X_ts[:split_index], X_ts[split_index:]
    X_narrative_train, X_narrative_test = X_narrative[:split_index], X_narrative[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    print(f"\nTraining Samples: {len(y_train)}, Testing Samples: {len(y_test)}")
    print(f"TS Input Shape (Train): {X_ts_train.shape}")
    print(f"Narrative Input Shape (Train): {X_narrative_train.shape}")

    # 3. Build and train the model
    ts_input_shape = (X_ts_train.shape[1], X_ts_train.shape[2])
    narrative_input_shape = (X_narrative_train.shape[1],)
    
    hybrid_model = build_hybrid_model(ts_input_shape, narrative_input_shape)
    hybrid_model.summary()
    
    print("\n--- 3. Training Hybrid Model ---")
    history = hybrid_model.fit(
        {'time_series_input': X_ts_train, 'narrative_embedding_input': X_narrative_train},
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=1,
        shuffle=False
    )

    # 4. Evaluate and Inverse Transform
    print("\n--- 4. Evaluating Hybrid Model Performance ---")
    
    # Predict on the test set
    y_pred_scaled = hybrid_model.predict({'time_series_input': X_ts_test, 'narrative_embedding_input': X_narrative_test})
    
    # Inverse transform to get the prediction in the original scale
    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler_target.inverse_transform(y_pred_scaled)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_pred_original - y_test_original))
    
    print(f"\n--- Holistic Horizon Hybrid Model Final Evaluation ---")
    print(f"Target Variable: {TARGET_COLUMN}")
    print(f"Test MAE (Emissions Intensity): {mae:.2f} kg CO₂ per MWh")
    print("\nNext Steps (Phase 4): Compare this MAE directly to the Phase 1 Baseline to quantify the value of the LLM narrative context.")
