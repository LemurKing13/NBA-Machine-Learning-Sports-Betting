import argparse
import re
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss


class BoosterWrapper:
    """Compatibility shim for legacy calibrator pickles."""

    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

ML_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    "OU-Cover",
    "OU",
]

UO_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    "OU-Cover",
]


def select_model_path(kind):
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def load_calibrator(model_path):
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        return None
    return joblib.load(calibration_path)


def season_from_date(date_value):
    year = date_value.year
    if date_value.month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    df["Season"] = df["Date"].map(season_from_date)
    return df


def evaluate_binary_or_multiclass(y_true, probabilities, labels):
    y_pred = np.argmax(probabilities, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate XGBoost models on a single NBA season.")
    parser.add_argument("--dataset", default="dataset_2012-26", help="Dataset table name.")
    parser.add_argument("--season", default="2024-25", help="Season label like 2024-25.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    season_df = df[df["Season"] == args.season].copy()

    if season_df.empty:
        raise ValueError(f"No rows found for season {args.season} in dataset {args.dataset}.")

    # Moneyline model
    ml_model_path = select_model_path("ML")
    ml_model = xgb.Booster()
    ml_model.load_model(str(ml_model_path))
    ml_calibrator = load_calibrator(ml_model_path)

    y_ml = season_df["Home-Team-Win"].astype(int).to_numpy()
    X_ml = season_df.drop(columns=ML_DROP_COLUMNS + ["Season"], errors="ignore").astype(float).to_numpy()
    if ml_calibrator is not None:
        p_ml = ml_calibrator.predict_proba(X_ml)
    else:
        p_ml = ml_model.predict(xgb.DMatrix(X_ml))
    ml_metrics = evaluate_binary_or_multiclass(y_ml, p_ml, labels=[0, 1])

    # Totals model
    uo_model_path = select_model_path("UO")
    uo_model = xgb.Booster()
    uo_model.load_model(str(uo_model_path))
    uo_calibrator = load_calibrator(uo_model_path)

    y_uo = season_df["OU-Cover"].astype(int).to_numpy()
    X_uo = season_df.drop(columns=UO_DROP_COLUMNS + ["Season"], errors="ignore").astype(float).to_numpy()
    if uo_calibrator is not None:
        p_uo = uo_calibrator.predict_proba(X_uo)
    else:
        p_uo = uo_model.predict(xgb.DMatrix(X_uo))
    uo_metrics = evaluate_binary_or_multiclass(y_uo, p_uo, labels=[0, 1, 2])

    print(f"Season: {args.season}")
    print(f"Games: {len(season_df)}")
    print("\nMoneyline (XGBoost)")
    print(f"Model: {ml_model_path.name}")
    print(f"Accuracy: {ml_metrics['accuracy']:.4f}")
    print(f"Log loss: {ml_metrics['log_loss']:.4f}")

    print("\nTotals (XGBoost)")
    print(f"Model: {uo_model_path.name}")
    print(f"Accuracy: {uo_metrics['accuracy']:.4f}")
    print(f"Log loss: {uo_metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()
