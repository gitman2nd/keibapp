from __future__ import annotations

from pathlib import Path
import pandas as pd


COURSE_MAP = {
    "1": "札幌",
    "2": "函館",
    "3": "福島",
    "4": "新潟",
    "4e": "新潟",
    "5": "東京",
    "6": "中山",
    "6e": "中山",
    "7": "中京",
    "8": "京都",
    "8e": "京都",
    "9": "阪神",
    "9e": "阪神",
    "10": "小倉",
}

WAKU_COLOR_MAP = {
    1: "#ffffff",   # 白
    2: "#1f1f1f",   # 黒
    3: "#e53935",   # 赤
    4: "#1e88e5",   # 青
    5: "#fdd835",   # 黄
    6: "#43a047",   # 緑
    7: "#fb8c00",   # オレンジ
    8: "#f48fb1",   # ピンク
}

SURFACE_COLOR_MAP = {
    "芝": "#2e7d32",
    "ダート": "#8d6e63",
}

TEXT_COLOR_ON_WAKU = {
    1: "#111111",
    2: "#ffffff",
    3: "#ffffff",
    4: "#ffffff",
    5: "#111111",
    6: "#ffffff",
    7: "#111111",
    8: "#111111",
}


def _normalize_course(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _safe_int(value, default=None):
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def load_race_data(
    data_path: str | Path = "data/ur_bunseki.parquet",
) -> pd.DataFrame:
    """
    parquet/csv のどちらでも読めるようにする。
    """
    path = Path(data_path)

    if not path.exists():
        alt_csv = path.with_suffix(".csv")
        if alt_csv.exists():
            path = alt_csv
        else:
            raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"未対応の拡張子です: {path.suffix}")

    return preprocess_race_data(df)


def preprocess_race_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 必須想定列
    expected_cols = [
        "レースid", "開催日", "コース", "R", "距離", "クラス", "出走頭数", "芝砂",
         "馬番", "馬名"
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    df["レースid"] = df["レースid"].astype(str)
    df["開催日"] = pd.to_datetime(df["開催日"], errors="coerce")

    if "年" not in df.columns:
        df["年"] = df["開催日"].dt.year
    if "月" not in df.columns:
        df["月"] = df["開催日"].dt.month
    if "月日" not in df.columns:
        df["月日"] = df["開催日"].dt.strftime("%m/%d")

    df["コースコード"] = df["コース"].apply(_normalize_course)
    df["コース名"] = df["コースコード"].map(COURSE_MAP).fillna(df["コース"].astype(str))

    df["R"] = pd.to_numeric(df["R"], errors="coerce").astype("Int64")
    df["距離"] = pd.to_numeric(df["距離"], errors="coerce").astype("Int64")
    df["出走頭数"] = pd.to_numeric(df["出走頭数"], errors="coerce").astype("Int64")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce").astype("Int64")

    df["芝砂"] = df["芝砂"].astype(str).str.strip()
    df["surface_color"] = df["芝砂"].map(SURFACE_COLOR_MAP).fillna("#9e9e9e")

    return df