import numpy as np
from torch_geometric_temporal.signal import StaticGraphTemporalSignal, DynamicGraphTemporalSignal

def build_static_temporal_signal(edge_index, edge_weight, X_array, y_array, num_timesteps_in):
    """
    Creates a StaticGraphTemporalSignal dataset required by recurrent GNNs.
    Features are reshaped to (Num Nodes, Seq Len * Features) for a static graph approach.
    """
    T = X_array.shape[0]
    X_list = []
    y_list = []
    for t in range(num_timesteps_in, T):
        # Slice history: [t-L, ..., t-1]
        past = X_array[t - num_timesteps_in:t]  
        
        # Transpose to (Node, Time, Feature)
        past_transposed = np.transpose(past, (1, 0, 2)) 
        
        # Flatten time and feature dimensions for static graph input (Node, L*F)
        node_features = past_transposed.reshape(past_transposed.shape[0], -1)
        
        X_list.append(node_features.astype(np.float32))
        y_list.append(y_array[t].astype(np.int64)) # Target label at time t
        
    return StaticGraphTemporalSignal(edge_index, edge_weight, X_list, y_list)


def build_dynamic_temporal_signal(X_list, edge_index_list, edge_weight_list, y_list, num_timesteps_in):
    """
    Creates a DynamicGraphTemporalSignal dataset where features and edge structure
    can change at every time step.
    """
    T = len(X_list)
    X_windows = []
    edge_idx_windows = []
    edge_weight_windows = []
    y_windows = []

    for t in range(num_timesteps_in, T):
        # Slice history: [t-L, ..., t-1]
        past_feats = np.stack(X_list[t-num_timesteps_in:t], axis=0)
        
        # Reshape to (Node, Seq Len * Features)
        past_feats = np.transpose(past_feats, (1, 0, 2))
        past_feats = past_feats.reshape(past_feats.shape[0], -1)
        
        X_windows.append(past_feats.astype(np.float32))
        
        # Edges corresponding to the target time t (t-L to t-1 are used implicitly by the model's update rules)
        # Note: PyG-T DynamicSignal expects the edge structures at index t to correspond to the target window.
        edge_idx_windows.append(edge_index_list[t].cpu().numpy())
        edge_weight_windows.append(edge_weight_list[t].cpu().numpy())

        y_windows.append(y_list[t].astype(np.int64))

    return DynamicGraphTemporalSignal(
        edge_idx_windows,
        edge_weight_windows,
        X_windows,
        y_windows
    )