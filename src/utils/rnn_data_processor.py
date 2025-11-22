import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def preprocess_data(df):
    """
    Sorts data, handles cyclic time encoding, scales numerical features, 
    one-hot encodes categorical features, and prepares the target scaler.
    """
    # Sort by station and date
    df = df.sort_values(['STATION_ID', 'DATE']).reset_index(drop=True)
    
    # Handle cyclic month
    df['MONTH_sin'] = np.sin(2 * np.pi * df['MONTH'] / 12.0)
    df['MONTH_cos'] = np.cos(2 * np.pi * df['MONTH'] / 12.0)
    
    # Define columns (matching original logic)
    non_normalized_cols = ['LATITUDE']
    normalized_cols = ['MAX', 'MIN', 'TEMP', 'Temperature']
    categorical_cols = ['TAG' , 'IS_AUGMENTED']
    
    # Scaler for numerical features
    scaler = MinMaxScaler()
    df[normalized_cols] = scaler.fit_transform(df[normalized_cols])
    
    # One-hot encode categorical columns
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = onehot_encoder.fit_transform(df[categorical_cols])
    encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=onehot_encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.reset_index(drop=True), encoded_categorical_df], axis=1)
    
    # Feature columns (includes one-hots, excludes time sin/cos to be added later)
    feature_cols = non_normalized_cols + normalized_cols + list(encoded_categorical_df.columns)
    
    # Convert cold_wave to int
    df['cold_wave'] = df['cold_wave'].astype(int)
    
    # Scaler for 'MIN' target (for regression inverse transform)
    min_scaler = MinMaxScaler()
    df['MIN_scaled'] = min_scaler.fit_transform(df[['MIN']])
    
    return df, feature_cols, scaler, min_scaler

def create_per_station_sequences(df, feature_cols, seq_len):
    """
    Generates time sequences (X) and corresponding targets (y_reg, y_cls) 
    independently for each station.
    """
    X_list, y_reg_list, y_cls_list = [], [], []
    
    current_feature_cols = [c for c in feature_cols if c in df.columns]
    
    for _, group in df.groupby('STATION_ID'):
        # Add encoded time features manually to features set (matching original logic)
        features = group[current_feature_cols + ['MONTH_sin', 'MONTH_cos']].values
        reg_targets = group['MIN_scaled'].values
        cls_targets = group['cold_wave'].values
        
        # Create sequences
        for i in range(len(group) - seq_len):
            X_list.append(features[i:i+seq_len])
            y_reg_list.append(reg_targets[i+seq_len])
            y_cls_list.append(cls_targets[i+seq_len])
            
    actual_input_size = X_list[0].shape[-1] if X_list else 0
    return np.array(X_list, dtype=np.float32), np.array(y_reg_list, dtype=np.float32), np.array(y_cls_list, dtype=np.float32), actual_input_size