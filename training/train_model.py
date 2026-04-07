from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_1to3"] = (df["着順"] <= 3).astype(int)
    return df


def get_feature_cols() -> list[str]:
    return [
        "単オッズ",
        "斤量",
        "出走間隔",
        "出走回数",
        "年齢",
    ]


def train_model(
    data_path: str | Path = "data/ur_bunseki.parquet",
    model_path: str | Path = "models/lgb_model.pkl",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    df = pd.read_parquet(data_path)
    df = create_features(df)
    df = df.dropna(subset=get_feature_cols() + ["is_1to3"])

    X = df[get_feature_cols()]
    y = df["is_1to3"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

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
        "seed": random_state,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, test_data],
        valid_names=["train", "test"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(
            {
                "model": model,
                "feature_cols": get_feature_cols(),
            },
            f,
        )

    results = {
        "model_path": str(model_path),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "positive_rate_train": y_train.mean(),
        "positive_rate_test": y_test.mean(),
        "best_iteration": model.best_iteration,
    }

    return results


if __name__ == "__main__":
    print("Training model...")
    results = train_model()
    print("\nTraining completed!")
    print(f"Model saved: {results['model_path']}")
    print(f"Train size: {results['train_size']}")
    print(f"Test size: {results['test_size']}")
    print(f"Positive rate (train): {results['positive_rate_train']:.3f}")
    print(f"Positive rate (test): {results['positive_rate_test']:.3f}")
    print(f"Best iteration: {results['best_iteration']}")
