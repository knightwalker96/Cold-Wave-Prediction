# Dynamic Spatio-Temporal Graph Neural Network for Cold Wave Prediction

This repository contains the source code and configuration used for the cold wave classification and minimum temperature regression modeling presented in the manuscript.

The core methodology involves utilizing recurrent Spatio-Temporal Graph Neural Networks (STGNNs) to model the non-Euclidean dependencies between geographically distributed weather stations over time.

-----

## 1\. Project Structure & Code Overview

The code is organized into the `src/` package to separate execution scripts from reusable components (models and utilities), ensuring high modularity and reproducibility.

| File/Module | Location | Functionality Implemented |
| :--- | :--- | :--- |
| **`__init__.py`** | `src/` | Package initializer. Allows Python to treat `src/` and its submodules (`models`, `utils`) as importable packages. |
| **`config.yaml`** | `PROJECT/` | **Primary configuration file.** Stores paths, hyperparameters, graph settings (e.g., K-neighbors), and feature definitions used across all experiments. |
| **`classification.py`** | `src/` | **Main GNN classification pipeline.** Loads dynamic graph data, runs GRU–SAGE model training with Focal Loss, and computes classification metrics. |
| **`regression.py`** | `src/` | **Main GNN regression pipeline.** Loads static/dynamic graph data, trains GRU–SAGE with SmoothL1Loss, and computes MAE/RMSE metrics. |
| **`graph_formulation.py`** | `src/` | **Graph dataset generator.** Runs preprocessing and invokes utilities to construct and save all static/dynamic PyG datasets as `.pt` files. |
| **`ml_baselines.py`** | `src/` | **Traditional ML baselines.** Builds lagged tabular datasets and evaluates Random Forest and Gradient Boosting using Time-Series Cross-Validation (TSCV). |
| **`pgt_baselines.py`** | `src/` | **PyG-Temporal baselines.** Runs benchmark experiments using wrappers of DCRNN, GConvGRU, GCLSTM, and EvolveGCNH implemented in `models/st_gnn_wrappers.py`. |
| **`stationIndep_rnn.py`** | `src/` | **Station-independent RNN baseline.** Implements a dual-head GRU/LSTM that processes each station’s time series independently for regression and classification tasks. |
| **`models/GRUSAGE.py`** | `src/models/` | **Temporal GNN model.** Defines the GRU–SAGE architecture used in the main spatio-temporal classification and regression experiments. |
| **`models/rnn_models.py`** | `src/models/` | **Dual-head RNN architecture.** Implements the `RNNModel` (GRU/LSTM) with separate regression and classification heads. |
| **`models/st_gnn_wrappers.py`** | `src/models/` | **PyG-Temporal model wrappers.** Wraps canonical `pytorch_geometric_temporal` models (e.g., DCRNN, GConvGRU, GCLSTM, EvolveGCNH) into PyTorch modules for training pipelines. |
| **`utils/feature_engineering.py`** | `src/utils/` | **Preprocessing utilities.** Implements `load_and_preprocess` for scaling, cyclic time encoding, one-hot encoding, and feature construction. |
| **`src/utils/graph_knn_factory.py`** | `src/utils/` | Utility function used as a precursor to the file pgt_Signal_processor . |
| **`src/utils/graph_dynamic_factory.py`** | `src/utils/` | Contains the logic to construct graph dataset variations |
| **`utils/data_processing.py`** | `src/utils/` | **GNN data utilities.** Includes functions like `prepare_temporal_data` and `custom_collate` for building lagged graph sequences for recurrent GNNs. |
| **`utils/pgt_signal_processor.py`** | `src/utils/` | **Temporal signal conversion.** Converts feature and edge arrays into `DynamicGraphTemporalSignal` objects for PyG-Temporal models. |
| **`utils/ml_baselines_eval.py`** | `src/utils/` | **ML evaluation utilities.** Creates lagged tabular datasets and evaluates traditional ML models with TSCV. |
| **`utils/rnn_data_processor.py`** | `src/utils/` | **Station sequence builder.** Prepares fixed-length input sequences and targets for the station-independent dual-head RNN model. |



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

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE.txt) file for details.
