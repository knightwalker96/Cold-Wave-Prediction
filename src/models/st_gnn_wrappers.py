import torch.nn as nn
from torch_geometric_temporal.nn.recurrent import GConvGRU, GConvLSTM, DCRNN, EvolveGCNH

class DCRNNWrapper(nn.Module):
    """
    Wrapper for Diffusion Convolutional Recurrent Neural Network (DCRNN).
    Processes spatio-temporal data and outputs logits for cold wave classification.
    """
    def __init__(self, in_channels, out_channels, node_count):
        super().__init__()
        # K=2 is standard for DCRNN diffusion steps
        self.dcrnn = DCRNN(in_channels, 64, K=2)
        self.linear = nn.Linear(64, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (Num Nodes in Batch, In Channels/Features)
        h = self.dcrnn(x, edge_index, edge_weight)
        # h shape: (Num Nodes in Batch, 64) - the final hidden state
        out = self.linear(h)
        return out.squeeze(-1) # Squeeze to (Num Nodes) for binary classification logits


class GConvGRUWrapper(nn.Module):
    """
    Wrapper for Graph Convolutional Gated Recurrent Unit (GConvGRU).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.gru = GConvGRU(in_channels, 64, K=2)
        self.linear = nn.Linear(64, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # x shape: (Num Nodes in Batch, In Channels/Features)
        h = self.gru(x, edge_index, edge_weight)
        out = self.linear(h)
        return out.squeeze(-1)


class GConvLSTMWrapper(nn.Module):
    """
    Wrapper for Graph Convolutional Long Short-Term Memory (GConvLSTM).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.lstm = GConvLSTM(in_channels, 64, K=2)
        self.linear = nn.Linear(64, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        # Note: GConvLSTM returns (h, c), but the forward pass for PyG-T requires 
        # a single output (h) from the recurrent call, which the module handles internally.
        h = self.lstm(x, edge_index, edge_weight)
        out = self.linear(h)
        return out.squeeze(-1)


class EvolveGCNHWrapper(nn.Module):
    """
    Wrapper for EvolveGCN-H (Evolving GCN Hidden) model.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.evolve = EvolveGCNH(in_channels, 64)
        self.linear = nn.Linear(64, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        h = self.evolve(x, edge_index, edge_weight)
        out = self.linear(h)
        return out.squeeze(-1)