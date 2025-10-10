import argparse
import os
import pickle
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.models import Sequential
    TENSORFLOW_AVAILABLE = True
except Exception:
    tf = None
    Sequential = None
    LSTM = None
    Dense = None
    Dropout = None
    TENSORFLOW_AVAILABLE = False

from utils import clean_column_name, safe_str, TARGET_COLUMN, FEATURE_COLUMNS, SEQUENCE_LENGTH, EPOCHS, BATCH_SIZE


def load_and_preprocess_data(file_path):
    """Loads, cleans, and structures the cross-sectional data into a time series."""
    print("--- 1. Loading and Cleaning Data ---")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please ensure the file is in the correct directory.")
        return None, None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None, None

    # Standardize column names in the DataFrame using the shared cleaning function
    cleaned_columns = [clean_column_name(c) for c in df.columns]
    df.columns = cleaned_columns

    # If Company_ID exists (after cleaning), sort by it to stabilize order.
    sort_col = None
    if 'Company_ID' in df.columns:
        sort_col = 'Company_ID'
    elif 'CompanyID' in df.columns:
        sort_col = 'CompanyID'

    if sort_col:
        df_ts = df.sort_values(by=sort_col).reset_index(drop=True)
    else:
        # Keep existing order but reset index
        df_ts = df.reset_index(drop=True)

    # Apply the shared cleaning function to the configuration list
    cleaned_feature_cols = [clean_column_name(col) for col in FEATURE_COLUMNS]
    cleaned_target_col = clean_column_name(TARGET_COLUMN)

    # Locate the target column robustly: exact match, then fuzzy match (emissions + mwh), then any 'emissions' fallback
    target_column_in_df = None
    if cleaned_target_col in df_ts.columns:
        target_column_in_df = cleaned_target_col
    else:
        low_cols = [c.lower() for c in df_ts.columns]
        # Prefer columns that contain both 'emissions' and 'mwh'
        for c in df_ts.columns:
            cl = c.lower()
            if 'emissions' in cl and 'mwh' in cl:
                target_column_in_df = c
                break
        # Next prefer any column that contains 'emissions'
        if target_column_in_df is None:
            for c in df_ts.columns:
                if 'emissions' in c.lower():
                    target_column_in_df = c
                    break

    if target_column_in_df is None:
        # No suitable target found
        print(f"Error: could not find the target column similar to {cleaned_target_col} in the CSV.")
        print("Available columns:", df_ts.columns.tolist())
        return None, None

    cols_to_use = cleaned_feature_cols + [target_column_in_df]

    # Verify that required columns exist. Try simple fuzzy matching if exact names are missing.
    missing = [c for c in cols_to_use if c not in df_ts.columns]
    if missing:
        resolved = {}
        for want in missing:
            found = None
            want_s = want.lower().replace('_', '')
            for cand in df_ts.columns:
                if want.lower() == cand.lower():
                    found = cand
                    break
                cand_s = cand.lower().replace('_', '')
                if want_s in cand_s or cand_s in want_s:
                    found = cand
                    break
            if found:
                resolved[want] = found

        # Replace any resolved names in cols_to_use
        for i, c in enumerate(cols_to_use):
            if c in resolved:
                cols_to_use[i] = resolved[c]

        still_missing = [c for c in cols_to_use if c not in df_ts.columns]
        if still_missing:
            print(f"Error: could not find required columns in CSV: {still_missing}")
            print("Available columns:", df_ts.columns.tolist())
            return None, None

    # Select only the features and the target, and drop any rows with NaN values
    df_ts = df_ts[cols_to_use].dropna()

    # Basic sanity check for dataset size
    if df_ts.shape[0] <= SEQUENCE_LENGTH:
        print(f"Error: Not enough rows after cleaning ({df_ts.shape[0]}) for sequence length {SEQUENCE_LENGTH}.")
        return None, None

    print(f"Dataset size after cleaning: {df_ts.shape}")
    return df_ts, df_ts.columns


def create_sequences(data, sequence_length):
    """
    Creates sequences of features (X) and the corresponding target value (Y)
    for use in an LSTM model.
    X: [t-N, t-N+1, ..., t-1] (Sequence of historical features)
    Y: [t] (Target value at the next time step)
    """
    X, y = [], []
    for i in range(len(data) - sequence_length):
        # Extract features (all columns except the last one, which is the target)
        X.append(data[i:i + sequence_length, :-1])
        # Extract the target value at the next step (i + sequence_length)
        # Note: We are predicting the target for the step immediately *after* the sequence ends.
        y.append(data[i + sequence_length, -1])
    return np.array(X), np.array(y)


def save_artifacts(model, scaler_features, scaler_target, use_tf: bool, save_dir: str = "artifacts"):
    """Save model and scalers to disk. TF models saved with model.save(); sklearn objects with joblib/pickle."""
    os.makedirs(save_dir, exist_ok=True)
    if use_tf and model is not None:
        try:
            model_path = os.path.join(save_dir, "phase1_lstm_baseline")
            model.save(model_path)
            print(f"Saved TensorFlow model to: {model_path}")
        except Exception as e:
            print(f"Warning: failed to save TF model: {e}")
    else:
        if model is not None:
            try:
                rf_path = os.path.join(save_dir, "phase1_rf_baseline.joblib")
                joblib_dump(model, rf_path)
                print(f"Saved RandomForest model to: {rf_path}")
            except Exception as e:
                print(f"Warning: failed to save RF model: {e}")

    # Save scalers
    try:
        feat_path = os.path.join(save_dir, "scaler_features.pkl")
        targ_path = os.path.join(save_dir, "scaler_target.pkl")
        with open(feat_path, "wb") as f:
            pickle.dump(scaler_features, f)
        with open(targ_path, "wb") as f:
            pickle.dump(scaler_target, f)
        print(f"Saved scalers to: {feat_path}, {targ_path}")
    except Exception as e:
        print(f"Warning: failed to save scalers: {e}")


def build_and_train_baseline(df_ts, feature_names, epochs: int = None, batch_size: int = None, force_rf: bool = False, save_artifacts_flag: bool = False, save_dir: str = "artifacts"):
    """Scales data, creates sequences, and trains the LSTM baseline model."""
    if epochs is None:
        epochs = EPOCHS
    if batch_size is None:
        batch_size = BATCH_SIZE
    print("\n--- 2. Data Scaling and Sequence Creation ---")

    # The last column is the target (Emissions_Intensity_kg_CO2_per_MWh)
    data = df_ts.values

    # Initialize separate scalers for features and target
    scaler_features = MinMaxScaler()
    scaler_target = MinMaxScaler()

    # Scale the features (all columns except the last one)
    features_scaled = scaler_features.fit_transform(data[:, :-1])

    # Scale the target separately (the last column)
    target_scaled = scaler_target.fit_transform(data[:, -1].reshape(-1, 1))

    # Recombine scaled data for sequence creation
    scaled_data = np.hstack((features_scaled, target_scaled))

    X, y = create_sequences(scaled_data, SEQUENCE_LENGTH)

    if X.size == 0 or y.size == 0:
        print("Error: No sequences were created. Check SEQUENCE_LENGTH and dataset size.")
        return

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"X_train shape: {X_train.shape} (Samples, Time Steps, Features)")
    print(f"y_train shape: {y_train.shape}")

    # --- 3. Build LSTM Model (Baseline) ---
    print("\n--- 3. Building and Training LSTM Baseline ---")

    # Define the required input shape for the LSTM layers
    input_seq_shape = (X_train.shape[1], X_train.shape[2])

    # Set seeds for reproducibility
    np.random.seed(42)
    if TENSORFLOW_AVAILABLE and tf is not None:
        try:
            tf.random.set_seed(42)
        except Exception:
            pass

    use_tf_model = TENSORFLOW_AVAILABLE and (not force_rf) and (tf is not None) and (Sequential is not None)
    model = None

    if use_tf_model:
        # Using tf.keras.Input as the first layer to explicitly define shape,
        # which resolves the UserWarning when defining Sequential models.
        model = Sequential([
            tf.keras.Input(shape=input_seq_shape),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1)  # Output layer for the single target variable
        ], name="EPM_LSTM_Baseline")

        model.compile(optimizer='adam', loss='mse')
        model.summary()

        # Train the model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            shuffle=False,  # Important for time-series data
        )

        # Predict on the test set
        y_pred_scaled = model.predict(X_test)
    else:
        print("Using RandomForest baseline (forced or TF not available).")

        # Flatten time-series sequences for a tree-based model: (samples, time_steps * features)
        n_samples, t_steps, n_feats = X_train.shape
        X_train_flat = X_train.reshape((n_samples, t_steps * n_feats))
        X_test_flat = X_test.reshape((X_test.shape[0], t_steps * n_feats))

        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train_flat, y_train.ravel())
        model = rf
        y_pred_scaled = rf.predict(X_test_flat).reshape(-1, 1)

    # --- 4. Evaluate and Inverse Transform ---
    print("\n--- 4. Evaluating Baseline Performance ---")

    # Optionally save artifacts (model + scalers)
    if save_artifacts_flag:
        try:
            save_artifacts(model, scaler_features, scaler_target, use_tf_model, save_dir=save_dir)
        except Exception as e:
            print(f"Warning: failed to save artifacts: {e}")

    # Inverse transform to get the prediction in the original scale (kg CO2 per MWh)
    y_test_original = scaler_target.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler_target.inverse_transform(y_pred_scaled)

    # Calculate Mean Absolute Error (MAE) - a common time-series metric
    mae = np.mean(np.abs(y_pred_original - y_test_original))

    print("\nBaseline Model Evaluation:")
    print("Target Variable:", safe_str(TARGET_COLUMN))
    print(f"Sequence Length (Lookback): {SEQUENCE_LENGTH} steps")
    print(f"Test MAE (Emissions Intensity): {mae:.2f} kg CO2 per MWh")
    print(safe_str("This MAE value serves as the benchmark for the Hybrid Model (Phase 3)."))
