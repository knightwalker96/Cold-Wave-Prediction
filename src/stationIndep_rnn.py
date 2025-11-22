import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sys
import yaml
from sklearn.metrics import precision_score, recall_score, f1_score, mean_absolute_error
from models.rnn_models import RNNModel
from utils.rnn_data_processor import preprocess_data, create_per_station_sequences

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    PATH_CONFIG = config['paths']
    LSTM_CONFIG = config['lstm_params']
    TRAIN_CONFIG = config['training_params']
except FileNotFoundError:
    print("Error: config.yaml not found. Using internal defaults.")
    PATH_CONFIG = {'part_1_path': 'data/datasets/part_1.csv'}
    LSTM_CONFIG = {'seq_len': 30, 'hidden_size': 64, 'num_layers': 2, 'pos_weight': 7.50}
    TRAIN_CONFIG = {'epochs': 50, 'learning_rate': 3e-4, 'batch_size_classification': 32}

# --- Constants ---
PART1_CSV = PATH_CONFIG.get('part_1_path')
SEQ_LEN = LSTM_CONFIG['seq_len']
#EPOCHS = TRAIN_CONFIG['epochs']
EPOCHS = 50
LR = TRAIN_CONFIG['learning_rate']
BATCH_SIZE = TRAIN_CONFIG.get('batch_size_classification', 32)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


try:
    part_1 = pd.read_csv(PART1_CSV)
except FileNotFoundError:
    print(f"Error: Could not find {PART1_CSV}. Ensure data is preprocessed and path is correct.")
    sys.exit(1)


def train_model(model, train_loader, epochs, lr, device):
    """
    Trains the dual-head RNN model using a combined MSE (regression) + BCEWithLogitsLoss (classification) loss.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reg_criterion = nn.MSELoss()
    
    # Positive weight for highly imbalanced classification task
    pos_weight = torch.tensor([LSTM_CONFIG['pos_weight']]).to(device)
    cls_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=12, gamma=0.35)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_reg_batch, y_cls_batch in train_loader:
            X_batch, y_reg_batch = X_batch.to(device), y_reg_batch.to(device).unsqueeze(1)
            y_cls_batch = y_cls_batch.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            reg_pred, cls_pred_logits = model(X_batch)
            
            reg_loss = reg_criterion(reg_pred, y_reg_batch)
            cls_loss = cls_criterion(cls_pred_logits, y_cls_batch)
            loss = reg_loss + cls_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f'[LSTM] Epoch {epoch+1:02d}/{epochs}, Loss: {total_loss / len(train_loader):.4f}')


def test_model(model, test_loader, min_scaler, device, threshold=0.5):
    """
    Evaluates the model on the test set, calculating and printing only classification metrics.
    """
    model.to(device)
    model.eval()
    cls_preds, cls_trues = [], []
    
    with torch.no_grad():
        for X_batch, y_reg_batch, y_cls_batch in test_loader:
            X_batch = X_batch.to(device)
            _, cls_pred_logits = model(X_batch)
            
            # Apply sigmoid to logits to get probabilities
            cls_preds.append(torch.sigmoid(cls_pred_logits).cpu().numpy()) 
            cls_trues.append(y_cls_batch.numpy())
    
    cls_preds = np.concatenate(cls_preds).flatten()
    cls_trues = np.concatenate(cls_trues).flatten()
    
    # Classification Metrics
    cls_preds_bin = (cls_preds >= threshold).astype(int)
    
    precision = precision_score(cls_trues, cls_preds_bin, zero_division=0)
    recall = recall_score(cls_trues, cls_preds_bin, zero_division=0)
    f1 = f1_score(cls_trues, cls_preds_bin, zero_division=0)

    print("\n--- Station Independent LSTM Classification Results ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")


if __name__ == "__main__":
    print(f"Starting Station-Independent LSTM training on {DEVICE}...")
    
    #  Preprocessing and Sequence Generation
    df, feature_cols, scaler, min_scaler = preprocess_data(part_1)
    
    X, y_reg, y_cls, input_size = create_per_station_sequences(df, feature_cols, seq_len=SEQ_LEN)
    
    #  Data Splitting
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = train_test_split(X, y_reg, y_cls, test_size=0.20, random_state=42)
    
    #  PyTorch Data Loaders
    train_dataset = TensorDataset(
        torch.from_numpy(X_train), 
        torch.from_numpy(y_reg_train), 
        torch.from_numpy(y_cls_train)
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test), 
        torch.from_numpy(y_reg_test), 
        torch.from_numpy(y_cls_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    #  Model Initialization and Training
    lstm_model = RNNModel(
        input_size=input_size, 
        hidden_size=LSTM_CONFIG['hidden_size'], 
        num_layers=LSTM_CONFIG['num_layers'],
        rnn_type='LSTM'
    )
    print(f"Training LSTM model with input size: {input_size} and seq_len: {SEQ_LEN}...")
    train_model(lstm_model, train_loader, epochs=EPOCHS, lr=LR, device=DEVICE)

    #  Testing
    test_model(lstm_model, test_loader, min_scaler, device=DEVICE)