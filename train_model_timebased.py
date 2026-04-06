from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import lightgbm as lgb

from data_loader import load_race_data


FEATURE_COLS_V1 = ["単オッズ", "斤量", "出走間隔", "出走回数", "年齢"]
FEATURE_COLS_V2 = ["斤量", "年齢", "出走間隔", "出走回数", "テン1F", "テン2F", "上がり2f", "馬場指数"]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_1to3"] = (df["着順"] <= 3).astype(int)
    return df


def get_latest_year(df: pd.DataFrame) -> int:
    return int(df["年"].max())


def train_timebased(
    data_path: str | Path = "data/ur_bunseki.parquet",
    model_dir: str | Path = "models",
    val_year: int | None = None,
) -> dict:
    df = load_race_data(data_path)
    df = create_features(df)

    latest_year = get_latest_year(df)
    if val_year is None:
        val_year = latest_year

    train_df = df[df["年"] < val_year].copy()
    val_df = df[df["年"] == val_year].copy()

    print(f"学習期間: {train_df['年'].min()} - {train_df['年'].max()}")
    print(f"検証期間: {val_year}")
    print(f"学習データ: {len(train_df):,}件")
    print(f"検証データ: {len(val_df):,}件")

    results = {}

    for version, feature_cols in [("v1", FEATURE_COLS_V1), ("v2", FEATURE_COLS_V2)]:
        print(f"\n{'='*50}")
        print(f"Training Model {version}")
        print(f"Features: {feature_cols}")
        print(f"{'='*50}")

        train_clean = train_df.dropna(subset=feature_cols + ["is_1to3"])
        val_clean = val_df.dropna(subset=feature_cols + ["is_1to3"])

        X_train = train_clean[feature_cols]
        y_train = train_clean["is_1to3"]
        X_val = val_clean[feature_cols]
        y_val = val_clean["is_1to3"]

        print(f"学習データ使用可能: {len(X_train):,}")
        print(f"検証データ使用可能: {len(X_val):,}")
        print(f"正例率 (train): {y_train.mean():.3f}")
        print(f"正例率 (val):   {y_val.mean():.3f}")

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        params = {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            valid_names=["train", "val"],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
        )

        model_path = Path(model_dir) / f"lgb_model_{version}_timebased.pkl"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": model,
                    "feature_cols": feature_cols,
                    "version": f"{version}_timebased",
                    "train_year_range": f"{train_df['年'].min()}-{train_df['年'].max()}",
                    "val_year": val_year,
                    "description": f"単オディズ含む" if version == "v1" else "単オディズ除外",
                },
                f,
            )

        y_pred_proba = model.predict(X_val)
        y_pred = (y_pred_proba >= 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        auc = roc_auc_score(y_val, y_pred_proba)

        print(f"\n{version} Results:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  AUC:       {auc:.4f}")
        print(f"  Best iteration: {model.best_iteration}")

        results[version] = {
            "model_path": str(model_path),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "best_iteration": model.best_iteration,
            "feature_importance": dict(zip(feature_cols, model.feature_importance())),
        }

    return results


if __name__ == "__main__":
    print("Training models with time-based split...")
    print("="*60)
    results = train_timebased()
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for version, result in results.items():
        print(f"\nModel {version}:")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
