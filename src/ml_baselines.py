import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from utils.ml_baselines_eval import create_lagged_dataset, evaluate_regression, evaluate_classification

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    path_config = config['paths']
    feature_config = config['feature_configs']
    train_config = config['training_params']
    
    data_path_p1 = path_config['part_1_path']
    data_path_p2 = path_config['part_2_path']
    ml_features = feature_config['model_features']
    n_splits = train_config['tscv_splits']
    
    # --- Data Loading ---
    part_1 = pd.read_csv(data_path_p1)
    part_2 = pd.read_csv(data_path_p2)
    
    # --- Model Definitions ---
    regression_models = {
        'RandomForest': RandomForestRegressor(random_state=42),
        'SVR': SVR(),
        'DecisionTree': DecisionTreeRegressor(random_state=42),
        'GradientBoosting': GradientBoostingRegressor(random_state=42)
    }
    classification_models = {
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42),
        'SVC': SVC(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    # Create Lagged Datasets 
    regression_datasets = {
        'part_1_lag1': create_lagged_dataset(part_1, ml_features, 'MIN', lags=1),
        #'part_1_lag2': create_lagged_dataset(part_1, ml_features, 'MIN', lags=2),
        #'part_1_lag3': create_lagged_dataset(part_1, ml_features, 'MIN', lags=3),
        #'part_2_lag1': create_lagged_dataset(part_2, ml_features, 'MIN', lags=1),
        #'part_2_lag2': create_lagged_dataset(part_2, ml_features, 'MIN', lags=2),
        #'part_2_lag3': create_lagged_dataset(part_2, ml_features, 'MIN', lags=3)
    }

    classification_datasets = {
        'part_1_lag1': create_lagged_dataset(part_1, ml_features, 'cold_wave', lags=1),
        #'part_1_lag2': create_lagged_dataset(part_1, ml_features, 'cold_wave', lags=2),
        #'part_1_lag3': create_lagged_dataset(part_1, ml_features, 'cold_wave', lags=3),
        #'part_2_lag1': create_lagged_dataset(part_2, ml_features, 'cold_wave', lags=1),
        #'part_2_lag2': create_lagged_dataset(part_2, ml_features, 'cold_wave', lags=2),
        #'part_2_lag3': create_lagged_dataset(part_2, ml_features, 'cold_wave', lags=3)
    }

    regression_results = {}
    classification_results = {}

    # --- Evaluation ---
    print("--- Evaluating Regression Baselines ---")
    for name, df in regression_datasets.items():
        regression_results[name] = evaluate_regression(df, 'MIN', regression_models, n_splits)

    print("--- Evaluating Classification Baselines ---")
    for name, df in classification_datasets.items():
        classification_results[name] = evaluate_classification(df, 'cold_wave', classification_models, n_splits)

    # --- Combine and Report Results ---
    
    # Combine regression results
    regression_combined = pd.DataFrame({'Model': list(regression_models.keys())})
    for name, result in regression_results.items():
        regression_combined = regression_combined.merge(
            result[['Model', 'MAE']].rename(columns={'MAE': f'MAE_{name}'}), 
            on='Model', 
            how='left'
        )

    # Combine classification results
    classification_combined = pd.DataFrame({'Model': list(classification_models.keys())})
    for name, result in classification_results.items():
        classification_combined = classification_combined.merge(
            result[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].rename(
                columns={
                    'Accuracy': f'Accuracy_{name}',
                    'Precision': f'Precision_{name}',
                    'Recall': f'Recall_{name}',
                    'F1 Score': f'F1_{name}'
                }
            ),
            on='Model',
            how='left'
        )

    print("\nRegression Results (MAE):")
    print(regression_combined)
    print("\nClassification Results:")
    print(classification_combined)

if __name__ == "__main__":
    main()