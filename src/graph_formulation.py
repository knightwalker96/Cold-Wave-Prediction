import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import os
import yaml
from utils.feature_engineering import load_and_preprocess
from utils.graph_dynamic_factory import construct_static_graph, construct_dynamic_graph

def main():
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    feature_config = config['feature_configs']
    path_config = config['paths']
    graph_config = config['graph_params']
    
    data_path_p1 = path_config['part_1_path']
    data_path_p2 = path_config['part_2_path']
    save_dir = path_config['dataset_variations_dir']

    os.makedirs(save_dir, exist_ok=True)
    
    datasets = {
        'small': data_path_p1,
        'large': data_path_p2
    }
    
    for name, path in datasets.items():
        for task in ['classification', 'regression']:
            print(f"Starting graph construction for {name} {task}...")
            df, f_cols, raw_m = load_and_preprocess(path, task, feature_config)
            
            k_static = graph_config[f'k_{name}_static']
            k_dynamic = graph_config[f'k_{name}_dynamic']
            
            static_graph = construct_static_graph(df, f_cols, raw_m, task, k_static)
            save_path_static = os.path.join(save_dir, f'static_{name}_{task}_k{k_static}.pt')
            torch.save(static_graph, save_path_static)
            print(f'Saved static {name} {task} graph to {save_path_static}')
            
            corr_feats = feature_config['correlation_features']
            dynamic_graph = construct_dynamic_graph(df, f_cols, raw_m, task, k_dynamic, corr_feats)
            save_path_dynamic = os.path.join(save_dir, f'dynamic_{name}_{task}_k{k_dynamic}.pt')
            torch.save(dynamic_graph, save_path_dynamic)
            print(f'Saved dynamic {name} {task} graph to {save_path_dynamic}')

if __name__ == "__main__":
    main()