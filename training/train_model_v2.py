from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


def get_feature_cols() -> list[str]:
    return [
        "斤量",
        "年齢",
        "出走間隔",
        "出走回数",
        "テン1F",
        "テン2F",
        "上がり2f",
        "馬場指数",
    ]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_1to3"] = (df["着順"] <= 3).astype(int)
    return df


def train_model_v2(
    data_path: str | Path = "data/ur_bunseki.parquet",
    model_path: str | Path = "models/lgb_model_v2.pkl",
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
                "version": "v2",
                "description": "Without odds feature",
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
        "feature_importance": dict(zip(get_feature_cols(), model.feature_importance())),
    }

    return results


if __name__ == "__main__":
    print("Training model v2 (without odds)...")
    results = train_model_v2()
    print("\nTraining completed!")
    print(f"Model saved: {results['model_path']}")
    print(f"Feature importance:")
    for name, imp in sorted(results['feature_importance'].items(), key=lambda x: -x[1]):
        print(f"  {name}: {imp}")
