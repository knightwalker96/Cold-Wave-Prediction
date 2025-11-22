import torch
import torch.nn as nn

class RNNModel(nn.Module):
    """
    An independent RNN model (LSTM or GRU) with a dual-head output for 
    simultaneous regression (MIN temperature) and classification (cold wave).
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, rnn_type='LSTM'):
        super().__init__()
        self.rnn_type = rnn_type
        
        # 1. Recurrent Layer
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
            
        # 2. Regression Head
        self.fc_reg = nn.Linear(hidden_size, 1)
        
        # 3. Classification Head (outputs logits for BCEWithLogitsLoss)
        self.fc_cls = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x shape: (Batch Size, Sequence Length, Input Size)
        
        # RNN Forward pass
        if self.rnn_type == 'LSTM':
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
            
        # Take the hidden state output from the last sequence step
        out = out[:, -1, :] # Shape: (Batch Size, Hidden Size)
        
        # Dual prediction
        reg = self.fc_reg(out)
        cls_logits = self.fc_cls(out)
        
        # reg is scaled temperature (regression), cls_logits are classification logits
        return reg, cls_logits