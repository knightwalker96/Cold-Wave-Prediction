import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from models.GRUSAGE import TemporalGNN_Regression 
from utils.data_processing import prepare_temporal_data, custom_collate

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

path_config = config['paths']
model_config = config['model_params']
train_config = config['training_params']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

dataset_path = os.path.join(path_config['dataset_variations_dir'], "static_large_regression_k7.pt")
try:
    gg = torch.load(dataset_path, weights_only=False)
    graph_dataset_list = list(gg.values())
except FileNotFoundError:
    print(f"Error: Dataset not found at {dataset_path}. Run graph_formulation.py first.")
    exit()

# ------------------------------------------------------------
# Main Training and Evaluation
# ------------------------------------------------------------
if __name__ == "__main__":
    num_time_steps = model_config['num_time_steps']
    
    # Data Preparation (using external functions)
    print("Preparing temporal sequences...", flush=True)
    temporal_graph_list = prepare_temporal_data(graph_dataset_list, num_time_steps, device)
    
    # Split Data
    train_split = int(len(temporal_graph_list) * 0.8)
    train_graphs = temporal_graph_list[:train_split]
    test_graphs = temporal_graph_list[train_split:]
    
    # DataLoaders using the custom collate function
    train_loader = DataLoader(
        train_graphs, batch_size=train_config['batch_size_regression'], shuffle=True, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_graphs, batch_size=train_config['batch_size_regression'], shuffle=False, collate_fn=custom_collate
    )
    
    # Model Initialization
    input_dim = train_graphs[0].x_time_series.shape[2]
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim_regression']
    
    # Initialize model using the modular class
    model = TemporalGNN_Regression(
        input_dim, 
        hidden_dim, 
        output_dim, 
        num_time_steps, 
        model_config['dropout_prob_regression']
    ).to(device)
    
    # Training Setup
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    
    # Loss functions (SmoothL1Loss/MAE_FN used for backpropagation, MSELoss used for logging)
    loss_fn = nn.MSELoss()  
    mae_fn = nn.SmoothL1Loss() 
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0.001
    )
    epochs = train_config['epochs']
    print(f"Starting regression training for {epochs} epochs...", flush=True)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Forward pass
            preds = model(batch.x_time_series, batch.edge_index).squeeze()
            # Loss calculated using SmoothL1Loss (mae_fn)
            loss = mae_fn(preds, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss (SmoothL1): {total_loss / len(train_loader):.4f}",
                flush=True,
            )
            
    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    print("Starting evaluation...", flush=True)
    
    with torch.no_grad():
        for batch in test_loader:
            preds = model(batch.x_time_series, batch.edge_index).squeeze()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    # Calculate Final Metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    mse = mean_squared_error(all_labels, all_preds)
    mae = mean_absolute_error(all_labels, all_preds)
    rmse = np.sqrt(mse)
    
    print("-" * 30, flush=True)
    print(f"Test MSE: {mse:.4f}", flush=True)
    print(f"Test MAE: {mae:.4f}", flush=True)
    print(f"Test RMSE: {rmse:.4f}", flush=True)