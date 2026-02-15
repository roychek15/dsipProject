import argparse
import json
import os
import wandb
from datetime import datetime
from capston_polaris_v4 import *
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV

from train_clustering import KmeansCluster
from train_random_forest import RandomForest
from train_basic_model import BasicModel

TARGET_COL = "review_scores_rating"


def load_data(csv_path: str) :
    train = pd.read_csv(csv_path+"/train.csv")
    test = pd.read_csv(csv_path+"/test.csv")
    X_train = train.drop(columns=[TARGET_COL])
    y_train = train[TARGET_COL]
    X_test = test.drop(columns=[TARGET_COL])
    y_test = test[TARGET_COL]
    return X_train, y_train, X_test, y_test


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest on processed airbnb dataset")
    parser.add_argument("--csv-path", type=str, default="data", help="Path to processed CSV (output of preprocess.py)")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--models-dir", type=str, default="models", help="Where to save the trained model")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory to save predictions and metrics")

    args = parser.parse_args()

    ensure_dir(args.models_dir)
    ensure_dir(args.results_dir)

    # split the dataset
    X_train, y_train, X_test, y_test = load_data(args.csv_path)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.validation_size, random_state=args.seed)

    basic = BasicModel()
    basic_metrics = basic.train_and_evaluate(X_train, y_train, X_val, y_val, seed=args.seed)
    basic.save_model(args.models_dir, args.results_dir)

    kmeans = KmeansCluster()
    best_kmeans_model, kmeans_metrics = kmeans.train_and_evaluate(X_train, y_train, X_val, y_val, seed=args.seed)
    kmeans.save_model(args.models_dir, args.results_dir)

    rnd_forest = RandomForest()
    best_rnd_forest_model, rnd_frst_metrics = rnd_forest.train_and_evaluate(X_train, y_train, X_val, y_val, seed=args.seed)
    rnd_forest.save_model(args.models_dir, args.results_dir)

    print("basic_model - rmse:", basic_metrics.get("val_rmse"))
    print(f'best_kmeans_model - k: {best_kmeans_model.n_clusters}, rmse: {kmeans_metrics.get("val_rmse")}')
    print("best_rnd_forest_model - rmse:", rnd_frst_metrics.get("val_rmse"))
