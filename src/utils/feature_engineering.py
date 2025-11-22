import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_and_preprocess(part_path, task, feature_config):
    """
    Loads raw data, performs normalization, time encoding, and one-hot encoding
    based on the specified task (regression or classification).
    """
    df = pd.read_csv(part_path)
    df['DATE'] = pd.to_datetime(df['DATE'])
    
    normalized_cols = feature_config[f'{task}_normalized_cols']
    categorical_cols = feature_config[f'{task}_categorical_cols']
    
    raw_min = None
    if task == 'regression':
        # Preserve original MIN values for regression targets
        raw_min = df[['STATION_ID', 'DATE', 'MIN']].copy()
    
    # 1. Scaling
    scaler = MinMaxScaler()
    df[normalized_cols] = scaler.fit_transform(df[normalized_cols])
    
    # 2. Time Features (Cyclic Encoding)
    # Assumes 'MONTH' column exists in the input DataFrame
    df['MONTH_sin'] = np.sin(2 * np.pi * df['MONTH'] / 12)
    df['MONTH_cos'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    
    # 3. Categorical Encoding
    onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_categorical = onehot_encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_categorical, columns=onehot_encoder.get_feature_names_out(categorical_cols))
    
    # Combine original data with encoded features
    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    
    # 4. Define Final Features (matching original logic)
    feature_cols = ['LATITUDE'] + normalized_cols + list(encoded_df.columns)
    
    if task == 'classification':
        # Ensure target is integer type for classification models
        df['cold_wave'] = df['cold_wave'].astype(int)
        
    return df, feature_cols, raw_min