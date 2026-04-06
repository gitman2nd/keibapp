from __future__ import annotations

import pickle
from pathlib import Path
from typing import Literal

import pandas as pd


MODEL_PATHS = {
    "v1": "models/lgb_model.pkl",
    "v2": "models/lgb_model_v2.pkl",
}


def load_model(model_path: str | Path = "models/lgb_model.pkl"):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["feature_cols"]


def predict_race(
    df: pd.DataFrame,
    race_id: str,
    model_version: Literal["v1", "v2"] = "v1",
) -> pd.DataFrame:
    model_path = MODEL_PATHS.get(model_version, MODEL_PATHS["v1"])
    model, feature_cols = load_model(model_path)

    race_df = df[df["レースid"].astype(str) == str(race_id)].copy()
    if race_df.empty:
        return pd.DataFrame()

    race_df = race_df.dropna(subset=feature_cols)
    if race_df.empty:
        return pd.DataFrame()

    X = race_df[feature_cols]
    probabilities = model.predict(X)
    race_df["予測確率"] = probabilities

    result = race_df[["馬番", "馬名", "予測確率"]].copy()
    result["予測確率"] = (result["予測確率"] * 100).round(1)
    result = result.sort_values("予測確率", ascending=False).reset_index(drop=True)
    result.index += 1
    result.index.name = "予測順位"

    return result


def predict_race_with_details(
    df: pd.DataFrame,
    race_id: str,
    model_version: Literal["v1", "v2"] = "v1",
) -> tuple[dict, pd.DataFrame]:
    model_path = MODEL_PATHS.get(model_version, MODEL_PATHS["v1"])
    model, feature_cols = load_model(model_path)

    race_df = df[df["レースid"].astype(str) == str(race_id)].copy()
    if race_df.empty:
        return {}, pd.DataFrame()

    head_cols = ["レースid", "開催日", "コース名", "距離", "クラス", "芝砂", "R"]
    head = race_df.iloc[0][head_cols].to_dict()

    race_df = race_df.dropna(subset=feature_cols)
    if race_df.empty:
        return head, pd.DataFrame()

    X = race_df[feature_cols]
    probabilities = model.predict(X)
    race_df["予測確率"] = probabilities

    result_cols = ["馬番", "馬名", "予測確率", "単オッズ", "斤量", "騎手", "出走間隔"]
    result = race_df[result_cols].copy()
    result["予測確率"] = (result["予測確率"] * 100).round(1)
    result = result.sort_values("予測確率", ascending=False).reset_index(drop=True)
    result.index += 1
    result.index.name = "予測順位"

    return head, result


def get_model_info(model_version: Literal["v1", "v2"] = "v1") -> dict:
    model_path = MODEL_PATHS.get(model_version, MODEL_PATHS["v1"])
    _, feature_cols = load_model(model_path)
    return {
        "version": model_version,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    from data_loader import load_race_data
    from analysis import build_race_master

    df = load_race_data("data/ur_bunseki.parquet")
    race_master = build_race_master(df)

    if not race_master.empty:
        sample_race_id = race_master.iloc[0]["レースid"]
        print(f"Predicting race: {sample_race_id}")
        print("\n--- Model v1 (with odds) ---")
        result_v1 = predict_race(df, sample_race_id, "v1")
        print(result_v1)
        print("\n--- Model v2 (without odds) ---")
        result_v2 = predict_race(df, sample_race_id, "v2")
        print(result_v2)
