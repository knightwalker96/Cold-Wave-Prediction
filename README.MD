# Dynamic Spatio-Temporal Graph Neural Network for Cold Wave Prediction

This repository contains the source code and configuration used for the cold wave classification and minimum temperature regression modeling presented in the manuscript.

The core methodology involves utilizing recurrent Spatio-Temporal Graph Neural Networks (STGNNs) to model the non-Euclidean dependencies between geographically distributed weather stations over time.

-----

## 1\. Project Structure & Code Overview

The code is organized into the `src/` package to separate execution scripts from reusable components (models and utilities), ensuring high modularity and reproducibility.

| File/Module | Location | Functionality Implemented |
| :--- | :--- | :--- |
| **`config.yaml`** | `PROJECT/` | **Primary Configuration.** Stores all hyperparameters (epochs, learning rates), graph settings (K-neighbors), feature lists, and file paths. |
| **`Data Preprocessing and EDA.ipynb`** | `PROJECT/` | **Data Pipeline Source.** Contains the original steps to clean, normalize, and generate the final stable splits (`part_1.csv`, `part_2.csv`). |
| **`src/graph_formulation.py`** | `src/` | **Graph Generation Script.** Orchestrates data loading and calls graph factory utilities to create daily snapshot datasets (`.pt` files). |
| **`src/classification.py` / `regression.py`** | `src/` | **Main STGNN Experiments.** Scripts for training and evaluating the core **GRU-SAGE** architecture (`models/temporal_gnn.py`) on their respective tasks. |
| **`src/ML_baselines.py`** | `src/` | **Traditional Baselines.** Runs time-series cross-validation (TSC-V) using standard models (Random Forest, Gradient Boosting). |
| **`src/pgt_baselines.py`** | `src/` | **PyG-Temporal Baselines.** Executes benchmarks using wrappers for established recurrent GNN models (DCRNN, GConvGRU, etc.). |
| **`src/models/temporal_gnn.py`** | `src/models/` | **Custom Architecture.** Defines the core Spatio-Temporal GNN (GRU temporal encoder + GraphSAGE spatial layers). |
| **`src/utils/graph_knn_factory.py`** | `src/utils/` | **Static Graph Logic.** Functions for calculating geodesic distances and building k-Nearest Neighbor (kNN) graph structures. |
| **`src/utils/graph_dynamic_factory.py`** | `src/utils/` | **Dynamic Graph Logic.** Functions for calculating daily inter-station feature correlations to create dynamic edge weights. |
| **`src/utils/pgt_signal_processor.py`** | `src/utils/` | **Signal Conversion.** Converts raw NumPy arrays into `pytorch_geometric_temporal` signal objects required by the baseline models. |


##  2. Logical Flow and Execution

The project pipeline is executed sequentially to ensure data dependencies are met:

1.  **Data Preparation (External):** Process the raw data (using the notebook) to generate stable files:
      * `data/processed/part_1.csv`
      * `data/processed/part_2.csv`
2.  **Graph Generation:** Transform the time series data into daily graph snapshots.
3.  **Model Training:** Run specific GNN or ML baseline experiments.

### Execution Steps

All executable scripts must be run using the **Python module system** (`python3 src.filename`) from the project's **root directory** (`PROJECT/`).

| Step | Command | Description |
| :--- | :--- | :--- |
| **1. Generate Graphs** | `python3 src/graph_formulation.py` | Creates all static and dynamic `.pt` files in `data/processed/dataset_variations/`. **Must run first.** |
| **2. Run Custom GNNs** | `python3 src/classification.py` | Trains the core GRU-SAGE model for cold wave classification. |
| | `python3 src/regression.py` | Trains the core GRU-SAGE model for temperature regression. |
| **3. Run Baselines** | `python3 src/ML_baselines.py` | Executes traditional ML models (RF, GB) using time-series cross-validation. |
| | `python3 src/pgt_baselines.py` | Executes PyG-Temporal models (DCRNN, GConvGRU) for benchmarking. |

-----

##  3. Dependencies

### Environment Setup

1.  **Clone the repository.**
2.  **Install dependencies** using the provided file:
    ```bash
    pip install -r requirements.txt
    ```

### Data and Code Availability

The data and code supporting the findings of this research are permanently archived and publicly accessible on Zenodo under the following Digital Object Identifiers (DOIs):

1.  **Code Repository (Software):** This specific version of the code is archived as software.
    * **DOI:** 10.5281/zenodo.17681005
    * **Link:** https://doi.org/10.5281/zenodo.17681005

2.  **Processed Data (Dataset):** The stable, processed datasets (`part_1.csv`, `part_2.csv`) essential for reproducing the results are archived separately.
    * **DOI:** 10.5281/zenodo.17680585
    * **Link:** https://doi.org/10.5281/zenodo.17680585
