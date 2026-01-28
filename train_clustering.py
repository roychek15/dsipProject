import argparse
import os
import wandb
from capston_polaris_v4 import *
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
os.environ['WANDB_API_KEY'] = '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'

def load_data(csv_path: str) :
    train = pd.read_csv(csv_path+"/train.csv")
    test = pd.read_csv(csv_path+"/test.csv")
    X_train = train.drop(columns=[Y_COL])
    y_train = train[Y_COL]
    X_test = test.drop(columns=[Y_COL])
    y_test = test[Y_COL]
    return X_train, y_train, X_test, y_test

def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def train_and_evaluate(X_train: pd.DataFrame, y_train: pd.DataFrame,
                       X_test: pd.DataFrame, y_test: pd.DataFrame):
    imputer = SimpleImputer(strategy='median')
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

    best_inertia = 0
    for k in [2, 3, 4, 5, 6, 8]:
        with wandb.init(project="kmeans-comparison", config={"k": k}):
            km = KMeans(n_clusters=k, n_init="auto")
            km.fit(X_train)

            # Calculate scores
            inertia = km.inertia_
            if (best_inertia == 0 or inertia < best_inertia):
                best_inertia = inertia
                best_k = k
                best_model = km
            sil_score = silhouette_score(X_train, km.labels_)

            # Log everything to compare in the dashboard
            wandb.log({
                "n_clusters": k,
                "inertia": inertia,
                "silhouette_score": sil_score
            })

    return best_model


def save_model(model, model_out: str):
    ensure_dir(os.path.dirname(model_out) or ".")
    joblib.dump(model, model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForest on processed bike dataset")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data",
        help="Path to processed CSV (output of preprocess.py)",
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/kmeans_model.joblib",
        help="Where to save the trained model",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/kmeans",
        help="Directory to save predictions and metrics",
    )

    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_data(args.csv_path)
    model = train_and_evaluate(X_train, y_train, X_test, y_test)

    save_model(model=model, model_out=args.model_out)

    print("Saved best kmeans model to: ", args.model_out)
    print(f"best model is when k={model.n_clusters}")
