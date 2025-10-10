import pandas as pd
import numpy as np
import re
import random
try:
    import tensorflow as tf
    from tensorflow.keras import Model, Sequential
    from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
    from tensorflow.keras.optimizers.legacy import Adam
    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    Model = None
    Sequential = None
    Input = None
    LSTM = None
    Dense = None
    Dropout = None
    Concatenate = None
    Adam = None
    TENSORFLOW_AVAILABLE = False
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# --- Configuration (MUST match previous phases) ---
FILE_PATH = r"C:\Users\acer\OneDrive\Desktop\EPM Model\Enterprise_Sustainable Power Evaluation_Dataset.csv"
SEQUENCE_LENGTH = 10        # Historical steps for Time-Series input
EMBEDDING_DIMENSION = 384   # Dimensionality of the Narrative Embedding
BATCH_SIZE = 32
EPOCHS = 75                 # Consistent training length for fair comparison

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

# --- Helper function for robust column cleaning ---
def standardize_column_name_robust(col_name):
    """Aggressively cleans and standardizes a column name."""
    cleaned = col_name
    
    # 1. Handle known encoding/symbol issues (e.g., CO₂ -> CO2)
    cleaned = cleaned.replace('â\x82\x82', '2') 
    
    # 2. Replace all non-alphanumeric, non-space characters with an underscore
    cleaned = re.sub(r'[^A-Za-z0-9\s_]', '_', cleaned)
    
    # 3. Replace spaces with underscores
    cleaned = cleaned.replace(' ', '_')
    
    # 4. Collapse multiple underscores and strip leading/trailing ones
    cleaned = re.sub(r'_+', '_', cleaned).strip('_')
    
    # 5. FIX: Ensure the target column is mapped correctly due to unpredictable source reading
    if 'Emissions_Intensity_kg_CO_per_MWh' in cleaned:
        return TARGET_CLEANED
        
    return cleaned


# --- Data Preparation and Simulation ---

def generate_augmented_data(file_path):
    """
    Loads, cleans, simulates narratives/embeddings, scales data, and returns 
    the necessary components for sequence creation.
    """
    print("--- 1. Data Preparation and Simulation ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        return None, None, None

    # Apply the single, robust standardization function to all columns
    df.columns = [standardize_column_name_robust(col) for col in df.columns]

    # Convert the dataset into a single pseudo-time-series
    # Sort by Company_ID when present; otherwise keep current order
    if 'Company_ID' in df.columns:
        df_ts = df.sort_values(by='Company_ID').reset_index(drop=True)
    else:
        df_ts = df.reset_index(drop=True)

    # The list of columns to use is now based on the standardized names
    clean_feature_cols = [standardize_column_name_robust(col) for col in FEATURE_COLUMNS]

    # If Company_ID exists in the cleaned dataframe, include it; otherwise omit it
    required_cols = clean_feature_cols + [TARGET_CLEANED]
    if 'Company_ID' in df_ts.columns:
        required_cols = ['Company_ID'] + required_cols

    # Filter for the required columns (this will raise a helpful KeyError if something's still missing)
    missing = [c for c in required_cols if c not in df_ts.columns]
    if missing:
        print(f"Error: required columns missing after cleaning: {missing}")
        print("Available columns:", df_ts.columns.tolist())
        return None, None, None

    df_ts = df_ts[required_cols].dropna()

    # Identify the feature columns present in the final DataFrame
    ts_features = [col for col in df_ts.columns if col in clean_feature_cols]

    # Calculate changes for narrative simulation (using a generic proxy, as full simulation is complex)
    df_ts['Emissions_Change'] = df_ts[TARGET_CLEANED].diff().fillna(0)
    
    # Simulate embeddings
    np.random.seed(42)
    embeddings = np.random.rand(len(df_ts), EMBEDDING_DIMENSION).astype(np.float32)
    
    # Use efficient concatenation to add embedding columns (avoiding PerformanceWarning)
    embedding_cols = [f'Embedding_{j}' for j in range(EMBEDDING_DIMENSION)]
    embeddings_df = pd.DataFrame(embeddings, columns=embedding_cols, index=df_ts.index)
    df_ts = pd.concat([df_ts, embeddings_df], axis=1)

    # Drop non-feature columns
    df_ts = df_ts.drop(columns=[col for col in df_ts.columns if 'Change' in col] + ['Company_ID'])
    
    # Final data scaling and structuring
    data = df_ts.values
    num_ts_features = len(ts_features)
    
    # 1. Scale Features (all except target)
    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(data[:, :-1])
    
    # 2. Scale Target
    scaler_target = MinMaxScaler()
    target_scaled = scaler_target.fit_transform(data[:, -1].reshape(-1, 1))

    # Recombine scaled data
    scaled_data = np.hstack((features_scaled, target_scaled))
    
    return scaled_data, num_ts_features, scaler_target


def create_sequences(data, sequence_length, num_ts_features):
    """
    Creates sequences for the dual-input model (Hybrid) and single-input (Baseline).
    Returns X_ts, X_narrative, y.
    """
    X_ts, X_narrative, y = [], [], []
    num_total_features = data.shape[1] - 1 
    
    for i in range(len(data) - sequence_length):
        # A. Time-Series Input (Historical sequence of TS Features only)
        # Columns [0 to num_ts_features - 1]
        X_ts.append(data[i: i + sequence_length, :num_ts_features])
        
        # B. Narrative Input (Embedding vector at prediction step)
        # Columns [num_ts_features to num_total_features - 1]
        X_narrative.append(data[i + sequence_length, num_ts_features:num_total_features])
        
        # C. Target (Target value at prediction step)
        y.append(data[i + sequence_length, -1])

    return np.array(X_ts, dtype=np.float32), np.array(X_narrative, dtype=np.float32), np.array(y, dtype=np.float32)

def create_tf_datasets(X_ts_train, X_ts_val, X_ts_test, X_narrative_train, X_narrative_val, X_narrative_test, y_train, y_val, y_test, batch_size):
    """Creates optimized TensorFlow Dataset objects for training and testing, incorporating validation set."""
    
    # Baseline Dataset (Single Input: Time-Series only)
    ds_baseline_train = tf.data.Dataset.from_tensor_slices((X_ts_train, y_train))
    ds_baseline_val = tf.data.Dataset.from_tensor_slices((X_ts_val, y_val))
    ds_baseline_test = tf.data.Dataset.from_tensor_slices((X_ts_test, y_test))
    
    # Hybrid Dataset (Dual Input: Time-Series and Narrative)
    X_train_hybrid = {'time_series_input': X_ts_train, 'narrative_embedding_input': X_narrative_train}
    X_val_hybrid = {'time_series_input': X_ts_val, 'narrative_embedding_input': X_narrative_val}
    X_test_hybrid = {'time_series_input': X_ts_test, 'narrative_embedding_input': X_narrative_test}
    
    ds_hybrid_train = tf.data.Dataset.from_tensor_slices((X_train_hybrid, y_train))
    ds_hybrid_val = tf.data.Dataset.from_tensor_slices((X_val_hybrid, y_val))
    ds_hybrid_test = tf.data.Dataset.from_tensor_slices((X_test_hybrid, y_test))

    # Apply batching and prefetching for performance
    ds_baseline_train = ds_baseline_train.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_baseline_val = ds_baseline_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_baseline_test = ds_baseline_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    ds_hybrid_train = ds_hybrid_train.shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_hybrid_val = ds_hybrid_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    ds_hybrid_test = ds_hybrid_test.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return (ds_baseline_train, ds_baseline_val, ds_baseline_test), (ds_hybrid_train, ds_hybrid_val, ds_hybrid_test)


# --- Model Definitions ---

def build_baseline_model(ts_input_shape):
    """LSTM Baseline Model (Phase 1)."""
    model = Sequential([
        Input(shape=ts_input_shape, name='ts_baseline_input'), # Use Input layer to avoid warning
        LSTM(units=50, return_sequences=True, name='lstm_baseline_1'),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False, name='lstm_baseline_2'),
        Dropout(0.2),
        Dense(units=1, activation='linear', name='baseline_output')
    ], name="EPM_LSTM_Baseline")
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def build_hybrid_model(ts_input_shape, narrative_input_shape):
    """Hybrid Fusion Model (Phase 3)."""
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
    
    # 3. Fusion Layer
    fusion_layer = Concatenate(name='fusion_layer')([ts_output, narrative_output])
    
    # 4. Final Prediction Head
    dense_final_1 = Dense(16, activation='relu')(fusion_layer)
    output = Dense(1, activation='linear', name='emissions_prediction')(dense_final_1)
    
    model = Model(inputs=[ts_input, narrative_input], outputs=output, name='Holistic_Horizon_EPM_Hybrid')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    return model

# --- Evaluation Function ---

def evaluate_model(model, ds_test, y_test_original_scale, scaler_target, model_name):
    """Evaluates the model using the TensorFlow Dataset."""
    
    # Predict
    y_pred_scaled = model.predict(ds_test, verbose=0)
    
    # Inverse transform
    y_pred_original = scaler_target.inverse_transform(y_pred_scaled)
    
    # Calculate Mean Absolute Error (MAE)
    mae = np.mean(np.abs(y_pred_original - y_test_original_scale))
    
    print(f"\n--- Evaluation: {model_name} ---")
    print(f"Test MAE ({TARGET_COLUMN}): {mae:.4f} kg CO₂ per MWh")
    return mae


# --- Main Execution ---

if __name__ == "__main__":
    
    # 1. Prepare and structure data
    scaled_data, num_ts_features, scaler_target = generate_augmented_data(FILE_PATH)
    if scaled_data is None:
        exit()

    X_ts, X_narrative, y = create_sequences(scaled_data, SEQUENCE_LENGTH, num_ts_features)

    # 2. Split into train, validation, and test sets (consistent split)
    test_size_ratio = 0.2
    val_size_ratio = 0.1 # 10% of the training data
    
    # Split into Train + Val and Test
    split_index = int(len(X_ts) * (1 - test_size_ratio))
    X_ts_tv, X_ts_test = X_ts[:split_index], X_ts[split_index:]
    X_narrative_tv, X_narrative_test = X_narrative[:split_index], X_narrative[split_index:]
    y_tv, y_test = y[:split_index], y[split_index:]

    # Split Train + Val into Train and Val
    val_index = int(len(X_ts_tv) * (1 - val_size_ratio))
    X_ts_train, X_ts_val = X_ts_tv[:val_index], X_ts_tv[val_index:]
    X_narrative_train, X_narrative_val = X_narrative_tv[:val_index], X_narrative_tv[val_index:]
    y_train, y_val = y_tv[:val_index], y_tv[val_index:]
    
    # Get the test target values in the original scale for final evaluation 
    y_test_original_scale = scaler_target.inverse_transform(y_test.reshape(-1, 1))

    print(f"\nTraining Samples: {len(y_train)}, Validation Samples: {len(y_val)}, Testing Samples: {len(y_test)}")
    
    ts_input_shape = (X_ts_train.shape[1], X_ts_train.shape[2])
    narrative_input_shape = (X_narrative_train.shape[1],)
    
    # 3. Create TensorFlow Datasets (Optimization step)
    (ds_baseline_train, ds_baseline_val, ds_baseline_test), (ds_hybrid_train, ds_hybrid_val, ds_hybrid_test) = \
        create_tf_datasets(X_ts_train, X_ts_val, X_ts_test, X_narrative_train, X_narrative_val, X_narrative_test, y_train, y_val, y_test, BATCH_SIZE)
    
    # --- 4. Run Baseline Model ---
    
    baseline_model = build_baseline_model(ts_input_shape)
    print("\n--- Training LSTM Baseline Model ---")
    baseline_model.fit(
        ds_baseline_train,
        epochs=EPOCHS,
        validation_data=ds_baseline_val,
        verbose=0,
    )
    mae_baseline = evaluate_model(baseline_model, ds_baseline_test, y_test_original_scale, scaler_target, "LSTM Baseline Model")
    
    # --- 5. Run Hybrid Model ---

    hybrid_model = build_hybrid_model(ts_input_shape, narrative_input_shape)
    print("\n--- Training Holistic Horizon Hybrid Model ---")
    hybrid_model.fit(
        ds_hybrid_train,
        epochs=EPOCHS,
        validation_data=ds_hybrid_val,
        verbose=0,
    )
    mae_hybrid = evaluate_model(hybrid_model, ds_hybrid_test, y_test_original_scale, scaler_target, "Hybrid Fusion Model")
    
    # --- 6. Comparative Analysis ---
    
    print("\n" + "="*50)
    print("      Holistic Horizon EPM Project: Final Comparison")
    print("="*50)
    print(f"Target Metric: {TARGET_COLUMN}")
    print(f"1. Pure Time-Series (LSTM Baseline) MAE: {mae_baseline:.4f}")
    print(f"2. Multi-Modal Hybrid (Fusion) MAE:       {mae_hybrid:.4f}")
    
    if mae_hybrid < mae_baseline:
        improvement = ((mae_baseline - mae_hybrid) / mae_baseline) * 100
        print(f"\nConclusion: Hybrid model outperformed the Baseline by {improvement:.2f}%.")
        print("This suggests that the **qualitative narrative context** (simulated LLM embeddings) adds significant predictive power and reduces forecast error.")
    elif mae_hybrid > mae_baseline:
        decline = ((mae_hybrid - mae_baseline) / mae_baseline) * 100
        print(f"\nConclusion: Hybrid model underperformed the Baseline by {decline:.2f}%.")
        print("This suggests that the simulated narrative context might be introducing noise or that the current fusion architecture needs tuning.")
    else:
        print("\nConclusion: The models performed identically. Further tuning is required.")
    
    print("="*50)
