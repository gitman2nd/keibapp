from __future__ import annotations

import pandas as pd


def build_race_master(df: pd.DataFrame) -> pd.DataFrame:
    """
    1レース1行のマスタを作る
    """
    race_cols = [
        "レースid",
        "開催日",
        "年",
        "月",
        "月日",
        "コースコード",
        "コース名",
        "R",
        "距離",
        "クラス",
        "出走頭数",
        "芝砂",
        "surface_color",
    ]

    existing = [c for c in race_cols if c in df.columns]
    race_df = (
        df[existing]
        .drop_duplicates(subset=["レースid"])
        .sort_values(["開催日", "コース名", "R"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    return race_df


def get_latest_week_races(race_df: pd.DataFrame) -> pd.DataFrame:
    """
    年・月日ベースの最新1週間
    """
    if race_df.empty:
        return race_df.copy()

    max_date = race_df["開催日"].max()
    start_date = max_date - pd.Timedelta(days=6)

    out = race_df.loc[race_df["開催日"].between(start_date, max_date)].copy()
    out = out.sort_values(["開催日", "コース名", "R"], ascending=[False, True, True])
    return out


def filter_races(
    race_df: pd.DataFrame,
    years: list | None = None,
    months: list | None = None,
    surfaces: list | None = None,
    distances: list | None = None,
    classes: list | None = None,
    limit: int = 30,
) -> pd.DataFrame:
    out = race_df.copy()

    if years:
        out = out[out["年"].isin([int(x) for x in years])]
    if months:
        out = out[out["月"].isin([int(x) for x in months])]
    if surfaces:
        out = out[out["芝砂"].isin(surfaces)]
    if distances:
        out = out[out["距離"].isin([int(x) for x in distances])]
    if classes:
        out = out[out["クラス"].astype(str).isin([str(x) for x in classes])]

    out = out.sort_values(["開催日", "レースid"], ascending=[False, False]).head(limit)
    return out


def get_filter_options(race_df: pd.DataFrame) -> dict:
    return {
        "years": sorted(
            race_df["年"].dropna().astype(int).unique().tolist(), reverse=True
        ),
        "months": sorted(race_df["月"].dropna().astype(int).unique().tolist()),
        "surfaces": sorted(race_df["芝砂"].dropna().astype(str).unique().tolist()),
        "distances": sorted(race_df["距離"].dropna().astype(int).unique().tolist()),
        "classes": sorted(race_df["クラス"].dropna().astype(str).unique().tolist()),
    }


def get_race_detail(df: pd.DataFrame, race_id: str) -> tuple[dict, pd.DataFrame]:
    target = df[df["レースid"].astype(str) == str(race_id)].copy()
    if target.empty:
        return {}, target

    head = target.iloc[0][
        ["レースid", "開催日", "コース名", "距離", "クラス", "芝砂", "R"]
    ].to_dict()

    horses = target.sort_values(["馬番", "馬名"])[["馬番", "馬名"]].reset_index(
        drop=True
    )
    return head, horses
