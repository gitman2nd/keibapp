from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)

from data_loader import load_race_data


FEATURE_COLS_V1 = ["単オッズ", "斤量", "出走間隔", "出走回数", "年齢"]
FEATURE_COLS_V2 = ["斤量", "年齢", "出走間隔", "出走回数", "テン1F", "テン2F", "上がり2f", "馬場指数"]


def load_model(model_path: str | Path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]


def evaluate_random_split(
    data_path: str | Path = "data/ur_bunseki.parquet",
    model_path: str | Path = "models/lgb_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = load_race_data(data_path)
    df["is_1to3"] = (df["着順"] <= 3).astype(int)

    model, feature_cols = load_model(model_path)

    df = df.dropna(subset=feature_cols + ["is_1to3"])
    X = df[feature_cols]
    y = df["is_1to3"]

    from sklearn.model_selection import train_test_split

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary, zero_division=0)
    recall = recall_score(y_test, y_pred_binary, zero_division=0)

    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    cm = confusion_matrix(y_test, y_pred_binary)

    return {
        "split_type": "random",
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "auc": round(roc_auc, 4),
        "confusion_matrix": cm.tolist(),
        "test_size": len(y_test),
        "positive_rate": round(y_test.mean(), 4),
    }


def evaluate_timebased(
    data_path: str | Path = "data/ur_bunseki.parquet",
    model_dir: str | Path = "models",
    val_year: int | None = None,
) -> dict:
    df = load_race_data(data_path)
    df["is_1to3"] = (df["着順"] <= 3).astype(int)

    latest_year = int(df["年"].max())
    if val_year is None:
        val_year = latest_year

    train_df = df[df["年"] < val_year].copy()
    val_df = df[df["年"] == val_year].copy()

    results = {}

    for version, feature_cols in [("v1", FEATURE_COLS_V1), ("v2", FEATURE_COLS_V2)]:
        model_path = Path(model_dir) / f"lgb_model_{version}_timebased.pkl"
        if not model_path.exists():
            continue

        model, _ = load_model(model_path)

        val_clean = val_df.dropna(subset=feature_cols + ["is_1to3"])
        X_val = val_clean[feature_cols]
        y_val = val_clean["is_1to3"]

        y_pred = model.predict(X_val)
        y_pred_binary = (y_pred >= 0.5).astype(int)

        accuracy = accuracy_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary, zero_division=0)
        recall = recall_score(y_val, y_pred_binary, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_pred)

        results[version] = {
            "split_type": "timebased",
            "val_year": val_year,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "auc": round(roc_auc, 4),
            "test_size": len(y_val),
            "positive_rate": round(y_val.mean(), 4),
        }

    return results


def print_comparison_summary(random_results: dict, timebased_results: dict) -> None:
    print("=" * 70)
    print("Model Comparison: Random Split vs Time-Based Split")
    print("=" * 70)
    print(f"{'Model':<8} {'Split Type':<15} {'AUC':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10}")
    print("-" * 70)

    for version in ["v1", "v2"]:
        if version in random_results:
            r = random_results[version]
            print(f"Model {version:<4} {'random':<15} {r['auc']:<10.4f} {r['accuracy']:<10.4f} {r['precision']:<10.4f} {r['recall']:<10.4f}")

        if version in timebased_results:
            t = timebased_results[version]
            print(f"Model {version:<4} {'timebased':<15} {t['auc']:<10.4f} {t['accuracy']:<10.4f} {t['precision']:<10.4f} {t['recall']:<10.4f}")
        
        print()

    print("=" * 70)
    print("Note: Time-based split uses future data for validation (no data leakage)")
    print("      Random split may include future data in training set")
    print("=" * 70)


if __name__ == "__main__":
    print("Evaluating models...\n")

    print("1. Random Split Evaluation")
    print("-" * 40)
    random_results = {}
    for version, path in [("v1", "models/lgb_model.pkl"), ("v2", "models/lgb_model_v2.pkl")]:
        try:
            result = evaluate_random_split(model_path=path)
            result["version"] = version
            random_results[version] = result
            print(f"Model {version}: AUC={result['auc']:.4f}, Acc={result['accuracy']:.4f}")
        except Exception as e:
            print(f"Model {version}: Error - {e}")

    print("\n2. Time-Based Split Evaluation")
    print("-" * 40)
    timebased_results = evaluate_timebased()

    for version, result in timebased_results.items():
        print(f"Model {version}: AUC={result['auc']:.4f}, Acc={result['accuracy']:.4f} (val_year={result['val_year']})")

    print("\n3. Comparison Summary")
    print("-" * 40)
    print_comparison_summary(random_results, timebased_results)
