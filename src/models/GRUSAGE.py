import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class FocalLoss(nn.Module):
    """
    Implements the Focal Loss for binary classification, which aims to address
    class imbalance by down-weighting the contribution of easy (well-classified) examples.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha if alpha is not None else 0.5
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate standard Binary Cross-Entropy Loss with Logits
        BCE_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        
        # Calculate alpha_t: weighting factor for positive (alpha) and negative (1-alpha) classes
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        # Calculate Focal Loss
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)

class TemporalGNN_Base(nn.Module):
    """
    Base Spatio-Temporal Graph Neural Network architecture (GRU + SAGE)
    for both Classification and Regression.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_time_steps, dropout_prob):
        super(TemporalGNN_Base, self).__init__()
        self.num_time_steps = num_time_steps
        self.input_dim = input_dim
        
        #  Temporal Component: GRU processes the time series features for each node
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        #  Spatial Component: GraphSAGE layers
        self.gnn1 = SAGEConv(hidden_dim, hidden_dim)
        self.gnn2 = SAGEConv(hidden_dim, output_dim)
        
        # Regularization and Activation
        self.dropout = nn.Dropout(dropout_prob)
        self.relu = nn.ReLU()
        
    def _forward_base(self, x_time_series, edge_index):
        # Reshape: (Total Nodes in Batch, Time Steps, Input Dim)
        x_time_series = x_time_series.view(-1, self.num_time_steps, self.input_dim)
        gru_out, _ = self.gru(x_time_series)
        last_time_step_out = gru_out[:, -1, :] # Last hidden state
        
        # Spatial Aggregation (GraphSAGE)
        gnn_out = self.gnn1(last_time_step_out, edge_index) + last_time_step_out
        gnn_out = self.relu(gnn_out)
        gnn_out = self.dropout(gnn_out)
        gnn_out = self.gnn2(gnn_out, edge_index)
        return gnn_out

class TemporalGNN_Regression(TemporalGNN_Base):
    """Temporal GNN customized for the regression task."""
    def forward(self, x_time_series, edge_index):
        gnn_out = self._forward_base(x_time_series, edge_index)
        # Squeeze the last dimension to get a scalar output per node
        return gnn_out.squeeze(-1) 

class TemporalGNN_Classification(TemporalGNN_Base):
    """Temporal GNN customized for the binary classification task."""
    def forward(self, x_time_series, edge_index):
        gnn_out = self._forward_base(x_time_series, edge_index)
        # Output is logits (Total Nodes, 1) for BCEWithLogitsLoss/FocalLoss
        return gnn_out