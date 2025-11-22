import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score, precision_score, recall_score, f1_score

def create_lagged_dataset(df, features, label, lags=1):
    """
    Creates a lagged version of the input DataFrame for time series forecasting,
    ensuring that lags are only taken within the same STATION_ID.
    """
    df = df.sort_values(['STATION_ID', 'DATE'])
    lagged_dfs = []
    
    # Only include features present in the dataframe
    current_features = [f for f in features if f in df.columns and f not in ['NAME', 'DATE', 'STATION_ID']]
    
    for lag in range(1, lags + 1):
        lagged = df[current_features].shift(lag)
        lagged.columns = [f'{col}_lag{lag}' for col in lagged.columns]
        lagged_dfs.append(lagged)
        
    result = pd.concat([df[['STATION_ID', 'DATE', label]]] + lagged_dfs, axis=1)
    
    # Filter out rows where the lagged data crosses station boundaries
    result = result[result['STATION_ID'] == result['STATION_ID'].shift(lags)]
    result = result.dropna()
    return result

def evaluate_regression(df, label, models, n_splits):
    """
    Evaluates standard regression models using TimeSeriesSplit cross-validation.
    """
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Drop identifying columns and the target variable
    X = df.drop(['STATION_ID', 'DATE', label], axis=1)
    y = df[label]
    
    # Handle categorical features (like TAG, PRCP) via one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for name, model in models.items():
        mae_scores = []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            mae_scores.append(mae)
            
        results.append({
            'Model': name,
            'MAE': np.mean(mae_scores)
        })
    return pd.DataFrame(results)

def evaluate_classification(df, label, models, n_splits):
    """
    Evaluates standard classification models using TimeSeriesSplit cross-validation.
    """
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    X = df.drop(['STATION_ID', 'DATE', label], axis=1)
    y = df[label]
    
    # Handle categorical features
    X = pd.get_dummies(X, drop_first=True)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    for name, model in models.items():
        acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            acc_scores.append(accuracy_score(y_test, y_pred))
            prec_scores.append(precision_score(y_test, y_pred, average='binary', zero_division=0))
            rec_scores.append(recall_score(y_test, y_pred, average='binary', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='binary', zero_division=0))
            
        results.append({
            'Model': name,
            'Accuracy': np.mean(acc_scores),
            'Precision': np.mean(prec_scores),
            'Recall': np.mean(rec_scores),
            'F1 Score': np.mean(f1_scores)
        })
    return pd.DataFrame(results)