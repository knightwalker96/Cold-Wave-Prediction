import torch
from torch_geometric.data import Batch, Data

def prepare_temporal_data(graph_dataset_list, num_time_steps, device):
    """
    Transforms a list of daily graph snapshots into a list of temporal (lagged)
    graph data objects for recurrent GNN training.

    Each output data object will contain x_time_series (history) and y (target).
    """
    temporal_data = []
    # Start iterating only after enough history is available
    for i in range(num_time_steps, len(graph_dataset_list)):
        # Prepare history: x_time_series is a stack of features from t-L to t-1
        # The stack order is (Node, Time, Feature)
        x_time_series = torch.stack(
            [graph_dataset_list[i - j - 1].x for j in range(num_time_steps)], dim=1
        )
        
        # Clone the target snapshot (t) to carry metadata (edge_index, y)
        data = graph_dataset_list[i].clone()
        
        # Move relevant tensors to the designated device
        data.x_time_series = x_time_series.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.to(device)
        
        temporal_data.append(data)
    return temporal_data

def custom_collate(data_list):
    """
    Custom collate function for PyG DataLoader to handle the complex temporal input (x_time_series)
    and correctly batch the graph connectivity (edge_index).
    """
    # Create the standard PyG Batch object
    batch = Batch.from_data_list(data_list)
    
    # Infer batching dimensions
    num_nodes = data_list[0].x.shape[0]
    batch_size = len(data_list)
    
    #  Correcting edge_index for batching (Crucial for GNN layers)
    edge_index = batch.edge_index.clone()
    
    # Calculate node offsets: [0, num_nodes, 2*num_nodes, ...]
    batch_offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
    
    # Apply offsets to source (0) and target (1) nodes based on their graph index in the batch
    # batch.batch provides the graph index for every node in the flattened batch
    edge_index[0] += batch_offsets[batch.batch[edge_index[0]]]
    edge_index[1] += batch_offsets[batch.batch[edge_index[1]]]
    
    batch.edge_index = edge_index
    
    #  Cleanup: Remove the original 'x' attribute as the model uses 'x_time_series'
    del batch.x
    
    return batch