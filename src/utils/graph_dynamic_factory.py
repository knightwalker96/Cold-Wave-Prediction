import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from geopy.distance import geodesic


def construct_static_graph(df, feature_cols, raw_min, task, k):
    """
    Constructs a k-NN graph based on geodesic distance (static structure) and 
    prepares the daily snapshot dataset.
    """
    unique_stations = df[['STATION_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates().sort_values('STATION_ID').reset_index(drop=True)
    nodes = unique_stations['STATION_ID'].values
    num_nodes = len(nodes)
    
    #  Calculate Geodesic Distances
    distances = np.zeros((num_nodes, num_nodes))
    for i, node_i in enumerate(nodes):
        lat_i, lon_i = unique_stations.iloc[i][['LATITUDE', 'LONGITUDE']]
        if pd.isna(lat_i) or pd.isna(lon_i): continue 
        for j, node_j in enumerate(nodes):
            if i != j:
                lat_j, lon_j = unique_stations.iloc[j][['LATITUDE', 'LONGITUDE']]
                distances[i, j] = distances[j, i] = geodesic((lat_i, lon_i), (lat_j, lon_j)).km
    
    #  Build k-NN Edges
    edge_index_list, edge_attr_list = [], []
    for i in range(num_nodes):
        # Find k nearest neighbors (excluding distance=0 self-loop implicitly by slicing [:k] on distances)
        nearest_neighbors = np.argsort(distances[i])[:k]
        for neighbor in nearest_neighbors:
            if i != neighbor:
                edge_index_list.append([i, neighbor])
                # Edge weight is inverse distance
                edge_attr_list.append(1.0 / (distances[i, neighbor] + 1e-5)) 
            
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    
    #  Assemble Daily Graph Dataset
    graph_dataset = {}
    for date in df['DATE'].unique():
        daily_data = df[df['DATE'] == date]
        
        if len(daily_data) != num_nodes: continue
        daily_features = daily_data.set_index('STATION_ID').loc[nodes]
        node_features = daily_features[feature_cols].values
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Determine Target (y)
        if task == 'regression':
            targets = raw_min[raw_min['DATE'] == date].set_index('STATION_ID').loc[nodes]['MIN'].values
            y = torch.tensor(targets, dtype=torch.float)
        else: # Classification
            y = torch.tensor(daily_features['cold_wave'].values, dtype=torch.long)
            
        
        graph_dataset[date.strftime('%Y-%m-%d')] = Data(x=x, 
                                                       edge_index=edge_index, 
                                                       edge_attr=edge_attr, 
                                                       y=y)
        
    return graph_dataset

def construct_dynamic_graph(df, feature_cols, raw_min, task, k, corr_features):
    """
    Constructs a daily dynamic graph based on inter-station feature correlation.
    """
    unique_stations = df[['STATION_ID', 'LATITUDE', 'LONGITUDE']].drop_duplicates().sort_values('STATION_ID').reset_index(drop=True)
    nodes = unique_stations['STATION_ID'].values
    num_nodes = len(nodes)
    
    # Prepare correlation dataframe
    df_for_corr = df.copy()
    if raw_min is not None and 'MIN' in corr_features and 'MIN' not in df_for_corr.columns:
         df_for_corr = pd.merge(df_for_corr, raw_min, on=['STATION_ID', 'DATE'], how='left')
    
    # Factorize categorical correlation features
    for col in corr_features:
        if col in df_for_corr.columns and df_for_corr[col].dtype == 'object':
            df_for_corr[col] = pd.factorize(df_for_corr[col])[0]
   
    graph_dataset = {}
    node_map = {node_id: i for i, node_id in enumerate(nodes)}
    
    # Map node index for sorting stability
    df['node_idx'] = df['STATION_ID'].map(node_map)
    df_for_corr['node_idx'] = df_for_corr['STATION_ID'].map(node_map)
    
    for date, daily_data_full in df.groupby('DATE'):
        if len(daily_data_full) != num_nodes: continue
        
        daily_data = daily_data_full.set_index('node_idx').sort_index()
        
        #  Calculate Daily Correlation Matrix
        corr_data = df_for_corr[df_for_corr['DATE'] == date].set_index('node_idx').sort_index()[corr_features]
        corr_values = corr_data.values.astype(float)
        corr_matrix = np.corrcoef(corr_values, rowvar=True)
        corr_matrix = np.nan_to_num(corr_matrix, 0)
        
        #  Build Dynamic Edges (k-Max Positive Correlation)
        edge_index_list, edge_attr_list = [], []
        for i in range(num_nodes):
            correlations = corr_matrix[i]
            nearest_neighbors = np.argsort(-correlations)
            
            added_count = 0
            for neighbor in nearest_neighbors:
                if added_count >= k: break
                
                # Edge exists if positive correlation (weight > 0)
                if i != neighbor and correlations[neighbor] > 0:
                    edge_index_list.append([i, neighbor])
                    edge_attr_list.append(correlations[neighbor])
                    added_count += 1
        
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
        
        #  Assemble Daily Graph Data
        node_features = torch.tensor(daily_data[feature_cols].values, dtype=torch.float)
        
        if task == 'regression':
            targets = raw_min[raw_min['DATE'] == date].set_index('STATION_ID').loc[nodes]['MIN'].values
            y = torch.tensor(targets, dtype=torch.float)
        else:
            y = torch.tensor(daily_data['cold_wave'].values, dtype=torch.long)
            
        graph_dataset[date.strftime('%Y-%m-%d')] = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, y=y)
        
    return graph_dataset