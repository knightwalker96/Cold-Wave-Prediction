import numpy as np
import pandas as pd
import torch
from geopy.distance import geodesic
import os

def construct_static_graph(df, feature_cols, k=11):
    """
    Constructs a k-Nearest Neighbor graph based on geodesic distance and prepares
    time series feature and target arrays (X_list, y_list).
    """
    original_coords = df[['LATITUDE', 'LONGITUDE', 'STATION_ID']].drop_duplicates().reset_index(drop=True)
    nodes = original_coords['STATION_ID'].values
    num_nodes = len(nodes)

    coords = original_coords[['LATITUDE', 'LONGITUDE']].values
    distances = np.zeros((num_nodes, num_nodes), dtype=float)
    
    # Calculate pairwise geodesic distances
    for i in range(num_nodes):
        lat_i, lon_i = coords[i]
        for j in range(i+1, num_nodes):
            lat_j, lon_j = coords[j]
            d = geodesic((lat_i, lon_i), (lat_j, lon_j)).km
            distances[i, j] = distances[j, i] = d

    # Construct Edges (k-NN)
    edge_index_list = []
    edge_weight_list = []
    for i in range(num_nodes):
        # Find k+1 nearest neighbors (includes self)
        nearest = np.argsort(distances[i])[:k+1] 
        for n in nearest:
            if n == i:
                continue
            edge_index_list.append([i, n])
            # Weight is inverse distance (plus small epsilon for stability)
            edge_weight_list.append(1.0/(distances[i, n] + 1e-6))

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight_list, dtype=torch.float)

    # Prepare time series data (ordered by nodes)
    dates = sorted(df['DATE'].unique())
    X_list = []  
    y_list = []  
    for date in dates:
        daily = df[df['DATE'] == date].copy()
        daily_indexed = daily.set_index('STATION_ID')
        
        # Check for full coverage before proceeding
        if not set(nodes).issubset(set(daily_indexed.index)):
            continue
            
        ordered = daily_indexed.loc[nodes]
        X_list.append(ordered[feature_cols].values.astype(float))
        y_list.append(ordered['cold_wave'].values.astype(int))

    return edge_index, edge_weight, np.array(X_list), np.array(y_list), nodes, dates


def construct_dynamic_correlation_graphs(df, feature_cols, k=7, correlation_features=None):
    """
    Constructs dynamic graphs where edges are based on daily feature correlation
    between stations.
    """
    if correlation_features is None:
        correlation_features = ['MIN', 'Temperature', 'TAG']
    
    original_coords = df[['LATITUDE', 'LONGITUDE', 'STATION_ID']].drop_duplicates().reset_index(drop=True)
    nodes = original_coords['STATION_ID'].values
    dates = sorted(df['DATE'].unique())

    X_list = []
    edge_index_list = []
    edge_weight_list = []
    y_list = []

    for date in dates:
        daily = df[df['DATE'] == date].copy()
        daily_indexed = daily.set_index('STATION_ID')
        if not set(nodes).issubset(set(daily_indexed.index)):
            continue
        
        ordered = daily_indexed.loc[nodes]
        X_list.append(ordered[feature_cols].values.astype(float))
        y_list.append(ordered['cold_wave'].values.astype(int))
        
        # Calculate Correlation Matrix
        corr_data = ordered[correlation_features].astype(float).values
        if corr_data.shape[1] == 0 or np.all(np.isnan(corr_data)) or corr_data.shape[0] < 2:
            corr_matrix = np.zeros((len(nodes), len(nodes)))
        else:
            corr_matrix = np.corrcoef(corr_data, rowvar=True)
            corr_matrix = np.nan_to_num(corr_matrix, 0.0)

        # Construct Dynamic Edges (k-Max Positive Correlation)
        edges = []
        weights = []
        for i in range(len(nodes)):
            corr_row = corr_matrix[i]
            # Sort descending correlation, take top k+1
            sorted_idx = np.argsort(-corr_row)[:k+1] 
            for j in sorted_idx:
                if i == j:
                    continue # Skip self-loops
                if corr_row[j] > 0:
                    edges.append([i, j])
                    weights.append(float(corr_row[j]))
        
        # Handle case with no positive correlation edges (add tiny dummy edge)
        if len(edges) == 0:
            for i in range(len(nodes)):
                for j in range(len(nodes)):
                    if i == j: continue
                    edges.append([i, j])
                    weights.append(1e-6)

        edge_index_list.append(torch.tensor(edges, dtype=torch.long).t().contiguous())
        edge_weight_list.append(torch.tensor(weights, dtype=torch.float))

    return X_list, edge_index_list, edge_weight_list, y_list, nodes, dates