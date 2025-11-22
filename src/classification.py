import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from models.GRUSAGE import TemporalGNN_Regression, FocalLoss
from utils.data_processing import prepare_temporal_data, custom_collate
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
path_config = config['paths']
model_config = config['model_params']
train_config = config['training_params']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}", flush=True)

# Load dataset (Hardcoded path as per original logic)
dataset_path = os.path.join(path_config['dataset_variations_dir'], "dynamic_small_classification_k7.pt")
try:
    gg = torch.load(dataset_path, weights_only=False)
    graph_dataset_list = list(gg.values())
except FileNotFoundError:
    print(f"Error: Dataset not found at {dataset_path}. Run graph_formulation.py first.")
    exit()

if __name__ == "__main__":
    num_time_steps = model_config['num_time_steps']
    
    #  Data Preparation
    print("Preparing temporal sequences...", flush=True)
    temporal_graph_list = prepare_temporal_data(graph_dataset_list, num_time_steps, device)
    
    # Split Data
    train_split = int(len(temporal_graph_list) * 0.8)
    train_graphs = temporal_graph_list[:train_split]
    test_graphs = temporal_graph_list[train_split:]

    # Infer model dimensions
    input_dim = train_graphs[0].x_time_series.shape[2]
    hidden_dim = model_config['hidden_dim']
    output_dim = model_config['output_dim_classification']
    
    # DataLoaders using the custom collate function
    train_loader = DataLoader(
        train_graphs, batch_size=train_config['batch_size_classification'], shuffle=False, collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_graphs, batch_size=train_config['batch_size_classification'], shuffle=False, collate_fn=custom_collate
    )
    
    #  Model and Training Setup
    model = TemporalGNN_Regression(
        input_dim, hidden_dim, output_dim, num_time_steps, model_config['dropout_prob_classification']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    loss_fn = FocalLoss(
        alpha=train_config['focal_loss_alpha'], 
        gamma=train_config['focal_loss_gamma']
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=100, eta_min=0.001
    )
    epochs = train_config['epochs']
    print(f"Starting training for {epochs} epochs on {len(train_graphs)} samples...", flush=True)

    #  Training Loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            # Forward pass: x_time_series contains lagged features
            out = model(batch.x_time_series, batch.edge_index).squeeze()
            
            # Loss expects logits (out) and float targets (batch.y)
            loss = loss_fn(out, batch.y.float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}", flush=True)

    #  Evaluation
    model.eval()
    all_preds, all_labels = [], []
    print("Starting evaluation...", flush=True)
    
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch.x_time_series, batch.edge_index).squeeze()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
    # Calculate Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    print("-" * 30, flush=True)
    print(f"Test Accuracy: {accuracy:.4f}", flush=True)
    print(f"Test Precision (Class 0): {precision[0]:.4f}, (Class 1): {precision[1]:.4f}", flush=True)
    print(f"Test Recall (Class 0): {recall[0]:.4f}, (Class 1): {recall[1]:.4f}", flush=True)
    print(f"Test F1-score (Class 0): {f1[0]:.4f}, (Class 1): {f1[1]:.4f}", flush=True)