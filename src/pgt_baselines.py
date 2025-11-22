import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric_temporal.signal import temporal_signal_split
import yaml
from models.st_gnn_wrappers import GConvGRUWrapper, GConvLSTMWrapper, DCRNNWrapper, EvolveGCNHWrapper
from utils.graph_knn_factory import construct_static_graph, construct_dynamic_correlation_graphs
from utils.pgt_signal_processor import build_static_temporal_signal, build_dynamic_temporal_signal

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    PATH_CONFIG = config['paths']
    MODEL_CONFIG = config['model_params']
    GRAPH_CONFIG = config['graph_params']
    TRAIN_CONFIG = config['training_params']
except FileNotFoundError:
    print("Error: config.yaml not found. Using default internal values.")
    PATH_CONFIG = {'part_1_path': 'data/processed/part_1.csv'}
    MODEL_CONFIG = {'num_time_steps': 3}
    GRAPH_CONFIG = {'k_small_static': 11, 'k_small_dynamic': 7}
    TRAIN_CONFIG = {'epochs': 750, 'batch_size_classification': 16}


PART1_CSV = PATH_CONFIG.get('part_1_path', "datasets/part_1.csv")
K_NEIGHBORS_STATIC = GRAPH_CONFIG.get('k_small_static', 11)
K_NEIGHBORS_DYNAMIC = GRAPH_CONFIG.get('k_small_dynamic', 7)
NUM_TIMESTEPS_IN = MODEL_CONFIG.get('num_time_steps', 3) 
EPOCHS = TRAIN_CONFIG.get('epochs', 750)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_and_preprocess(part1_path):
    """Loads data and performs feature normalization and encoding."""
    part_1 = pd.read_csv(part1_path)
    df = part_1.copy()
    
    normalized_cols = ['MAX', 'MIN', 'Temperature']
    
    # Scale numerical features
    scaler = MinMaxScaler()
    for c in normalized_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[normalized_cols] = scaler.fit_transform(df[normalized_cols])
    
    # Cyclic time features (if MONTH exists)
    if 'MONTH' in df.columns:
        df['MONTH_sin'] = np.sin(2 * np.pi * df['MONTH'] / 12)
        df['MONTH_cos'] = np.cos(2 * np.pi * df['MONTH'] / 12)
    
    # One-hot encode categorical features
    categorical_cols = [c for c in ['TAG', 'IS_AUGMENTED'] if c in df.columns]
    if len(categorical_cols) > 0:
        onehot_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_categorical = onehot_encoder.fit_transform(df[categorical_cols].astype(str))
        encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=onehot_encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.reset_index(drop=True), encoded_categorical_df], axis=1)
    else:
        encoded_categorical_df = pd.DataFrame(index=df.index)

    # Define final features
    feature_cols = ['LATITUDE'] + normalized_cols + list(encoded_categorical_df.columns)
    
    df['cold_wave'] = df['cold_wave'].astype(int)
    
    # Ensure DATE column is datetime
    if not np.issubdtype(df['DATE'].dtype, np.datetime64):
        df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')

    df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    df = df.reset_index(drop=True)
    return df, feature_cols


def train_epoch_model(model, device, data_iterator, optimizer, loss_fn):
    """Performs one epoch of training."""
    model.train()
    total_loss = 0.0
    n_steps = 0
    for snapshot in data_iterator:
        # Load features (x) and graph structure (edge_index, edge_weight)
        x = torch.from_numpy(snapshot.x).to(device) if isinstance(snapshot.x, np.ndarray) else snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        edge_weight = getattr(snapshot, "edge_weight", None)
        if edge_weight is not None:
            edge_weight = edge_weight.to(device)
            
        y = torch.from_numpy(snapshot.y).to(device) if isinstance(snapshot.y, np.ndarray) else snapshot.y.to(device)

        optimizer.zero_grad()
        preds = model(x, edge_index, edge_weight)
        
        # Prepare targets and predictions for BCEWithLogitsLoss
        preds = preds.view(-1)  
        y = y.view(-1).float()  
        
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss.item())
        n_steps += 1
        
    return total_loss / max(1, n_steps)


def eval_model(model, device, data_iterator):
    """Evaluates the model and returns classification metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for snapshot in data_iterator:
            x = torch.from_numpy(snapshot.x).to(device) if isinstance(snapshot.x, np.ndarray) else snapshot.x.to(device)
            edge_index = snapshot.edge_index.to(device)
            edge_weight = getattr(snapshot, "edge_weight", None)
            if edge_weight is not None:
                edge_weight = edge_weight.to(device)
            y = torch.from_numpy(snapshot.y).to(device) if isinstance(snapshot.y, np.ndarray) else snapshot.y.to(device)

            logits = model(x, edge_index, edge_weight)
            
            # Classification
            probs = torch.sigmoid(logits.view(-1))
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.view(-1).cpu().numpy().tolist())

    if len(all_labels) == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    # Calculate binary classification metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=0)
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def run_all_models_on_data(df, feature_cols):
    
    # --- Data Preparation ---
    print("Building dynamic graph dataset (correlation-based edges)...")
    # Using dynamic graph construction and signal building
    X_dyn_list, edge_idx_list, edge_weight_list, y_dyn_list, nodes_dyn, dates_dyn = construct_dynamic_correlation_graphs(df, feature_cols, k=K_NEIGHBORS_DYNAMIC)
    print(f"Dynamic: {len(X_dyn_list)} snapshots, {len(nodes_dyn)} nodes, {X_dyn_list[0].shape[1]} features (per snapshot)")
    
    dyn_signal = build_dynamic_temporal_signal(X_dyn_list, edge_idx_list, edge_weight_list, y_dyn_list, NUM_TIMESTEPS_IN)

    print("Splitting dynamic signal into train/test (80/20)...")
    train_dyn, test_dyn = temporal_signal_split(dyn_signal, train_ratio=0.8)
    
    # Calculate input feature size for GNN wrappers: Time Steps * Features per Snapshot
    feature_size = X_dyn_list[0].shape[1]
    in_channels = feature_size * NUM_TIMESTEPS_IN 

    # --- Model Initialization ---
    models = {
        "GConvGRU": GConvGRUWrapper(in_channels=in_channels, out_channels=1).to(DEVICE),
        #"DCRNN": DCRNNWrapper(in_channels=in_channels, out_channels=1).to(DEVICE),
        #"GConvLSTM": GConvLSTMWrapper(in_channels=in_channels, out_channels=1).to(DEVICE),
        #"EvolveGCNH": EvolveGCNHWrapper(in_channels=in_channels, out_channels=1).to(DEVICE)
    }

    # Loss function for binary classification logits
    loss_fn = nn.BCEWithLogitsLoss()
    results = {}

    # --- Training Loop ---
    for name, model in models.items():
        print("\n" + "="*60)
        print(f"Training model: {name}")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(1, EPOCHS + 1):
            train_loss = train_epoch_model(model, DEVICE, train_dyn, optimizer, loss_fn)
            if epoch % 50 == 0:
                val_metrics = eval_model(model, DEVICE, test_dyn)
                print(f"[{name}] Epoch {epoch:03d} TrainLoss: {train_loss:.4f}  ValF1: {val_metrics.get('f1', float('nan')):.4f}  Acc: {val_metrics.get('accuracy', float('nan')):.4f}")
        
        # Final Evaluation
        test_metrics = eval_model(model, DEVICE, test_dyn)
        print(f"Final test metrics for {name}: {test_metrics}")
        results[name] = test_metrics

    return results

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df, feature_cols = load_and_preprocess(PART1_CSV)
    print(f"Feature columns used: {feature_cols}")

    results = run_all_models_on_data(df, feature_cols)
    print("\nALL RESULTS SUMMARY:")
    for k, v in results.items():
        print(k, v)