from __future__ import annotations

from dash import Dash, html, dcc, Input, Output, State, ALL, no_update
import dash
import logging

# ※これらは既存のファイルから正しくインポートされる必要があります
from data_loader import load_race_data
from analysis import (
    build_race_master,
    get_latest_week_races,
    filter_races,
    get_filter_options,
    get_race_detail,
)

# -------------------------
# データ読込
# -------------------------
DF = load_race_data("data/ur_bunseki.parquet")
RACE_MASTER = build_race_master(DF)
FILTER_OPTIONS = get_filter_options(RACE_MASTER)


# -------------------------
# アプリ設定
# -------------------------
logging.basicConfig(
    filename="dash_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


# -------------------------
# 共通UI (dcc.Linkを使用して遷移をシンプル化)
# -------------------------
def race_button_card(race_row, race_id):
    """最新レースグリッド用のリンクボタン"""
    text = f"{int(race_row['距離'])}m / {race_row['クラス']} / {int(race_row['出走頭数'])}頭"
    return dcc.Link(
        html.Button(
            text,
            style={
                "width": "100%",
                "padding": "8px 10px",
                "margin": "4px 0",
                "borderRadius": "10px",
                "border": f"2px solid {race_row['surface_color']}",
                "background": "#ffffff",
                "cursor": "pointer",
                "fontSize": "13px",
                "textAlign": "left",
            },
        ),
        href=f"/race/{race_id}",
    )


def search_race_button(race_row, race_id):
    """検索結果リスト用のリンクボタン"""
    text = f"{race_row['年']} / {race_row['月日']} / {int(race_row['距離'])}m / {race_row['クラス']}"
    return dcc.Link(
        html.Button(
            text,
            style={
                "width": "100%",
                "padding": "10px 12px",
                "marginBottom": "8px",
                "borderRadius": "10px",
                "border": f"2px solid {race_row['surface_color']}",
                "background": "#ffffff",
                "cursor": "pointer",
                "textAlign": "left",
            },
        ),
        href=f"/race/{race_id}",
    )


def build_top_grid(latest_df):
    courses = [
        "札幌",
        "函館",
        "福島",
        "新潟",
        "東京",
        "中山",
        "中京",
        "京都",
        "阪神",
        "小倉",
    ]
    grid_rows = []

    header = [html.Div("", style={"fontWeight": "bold"})] + [
        html.Div(c, style={"fontWeight": "bold", "textAlign": "center"})
        for c in courses
    ]
    grid_rows.extend(header)

    for r in range(1, 13):
        grid_rows.append(
            html.Div(
                f"{r}R",
                style={
                    "fontWeight": "bold",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "minHeight": "60px",
                },
            )
        )
        for course in courses:
            row = latest_df[(latest_df["コース名"] == course) & (latest_df["R"] == r)]
            if row.empty:
                grid_rows.append(
                    html.Div(
                        "",
                        style={
                            "border": "1px solid #e0e0e0",
                            "borderRadius": "10px",
                            "minHeight": "60px",
                            "background": "#fafafa",
                        },
                    )
                )
            else:
                race_row = row.iloc[0]
                grid_rows.append(race_button_card(race_row, race_row["レースid"]))

    return html.Div(
        grid_rows,
        style={
            "display": "grid",
            "gridTemplateColumns": "60px repeat(10, 1fr)",
            "gap": "8px",
            "marginBottom": "32px",
        },
    )


def build_top_page():
    latest_df = get_latest_week_races(RACE_MASTER)
    return html.Div(
        [
            html.H1("競馬アプリ"),
            html.P("最新1週間のレース"),
            build_top_grid(latest_df),
            html.H2("レース検索"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("年"),
                            dcc.Dropdown(
                                id="filter-year",
                                multi=True,
                                options=[
                                    {"label": str(x), "value": x}
                                    for x in FILTER_OPTIONS["years"]
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("月"),
                            dcc.Dropdown(
                                id="filter-month",
                                multi=True,
                                options=[
                                    {"label": str(x), "value": x}
                                    for x in FILTER_OPTIONS["months"]
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("芝砂"),
                            dcc.Dropdown(
                                id="filter-surface",
                                multi=True,
                                options=[
                                    {"label": str(x), "value": x}
                                    for x in FILTER_OPTIONS["surfaces"]
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("距離"),
                            dcc.Dropdown(
                                id="filter-distance",
                                multi=True,
                                options=[
                                    {"label": str(x), "value": x}
                                    for x in FILTER_OPTIONS["distances"]
                                ],
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("クラス"),
                            dcc.Dropdown(
                                id="filter-class",
                                multi=True,
                                options=[
                                    {"label": str(x), "value": x}
                                    for x in FILTER_OPTIONS["classes"]
                                ],
                            ),
                        ]
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(5, 1fr)",
                    "gap": "12px",
                    "marginBottom": "20px",
                },
            ),
            html.Div(id="search-result-area"),
        ]
    )


def build_race_page(race_id: str):
    head, horses = get_race_detail(DF, race_id)

    # TOPへ戻るボタンもdcc.Linkで定義
    back_link = dcc.Link(
        "← TOPへ",
        href="/",
        style={
            "display": "inline-block",
            "padding": "10px 20px",
            "marginBottom": "20px",
            "backgroundColor": "#eee",
            "borderRadius": "5px",
            "textDecoration": "none",
            "color": "#333",
        },
    )

    if not head:
        return html.Div([back_link, html.H2("レースが見つかりません")])

    horse_cards = []
    for _, row in horses.iterrows():
        horse_cards.append(
            html.Div(
                [
                    html.Div(
                        f"{int(row['馬番'])}",
                        style={
                            "width": "40px",
                            "fontWeight": "bold",
                            "textAlign": "center",
                        },
                    ),
                    html.Div(
                        row["馬名"],
                        style={
                            "flex": 1,
                            "paddingBottom": "6px",
                            "color": "#111111",
                            "fontWeight": "500",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "padding": "10px 0",
                },
            )
        )

    return html.Div(
        [
            back_link,
            html.H1(f"{head['コース名']} {int(head['R'])}R"),
            html.Div(
                [
                    html.Div(
                        f"開催日: {head['開催日'].strftime('%Y-%m-%d') if head['開催日'] is not None else ''}"
                    ),
                    html.Div(f"コース: {head['コース名']}"),
                    html.Div(f"距離: {int(head['距離'])}m"),
                    html.Div(f"クラス: {head['クラス']}"),
                    html.Div(f"芝砂: {head['芝砂']}"),
                    html.Div(f"レースID: {head['レースid']}"),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(2, 1fr)",
                    "gap": "8px",
                    "padding": "16px",
                    "background": "#fafafa",
                    "borderRadius": "12px",
                    "marginBottom": "24px",
                },
            ),
            html.H2("出走馬"),
            html.Div(horse_cards),
        ]
    )


# -------------------------
# レイアウト (Store等の複雑な遷移管理を削除)
# -------------------------
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
)


# -------------------------
# コールバック
# -------------------------


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    """URLが変更されたらページを切り替える"""
    logging.info(f"アクセスされたURL: {pathname}")
    if pathname and pathname.startswith("/race/"):
        race_id = pathname.split("/race/")[-1]
        return build_race_page(race_id)
    return build_top_page()


@app.callback(
    Output("search-result-area", "children"),
    Input("filter-year", "value"),
    Input("filter-month", "value"),
    Input("filter-surface", "value"),
    Input("filter-distance", "value"),
    Input("filter-class", "value"),
)
def update_search_results(years, months, surfaces, distances, classes):
    """検索条件に合わせて結果を更新"""
    result = filter_races(
        RACE_MASTER,
        years=years,
        months=months,
        surfaces=surfaces,
        distances=distances,
        classes=classes,
        limit=30,
    )

    if result.empty:
        return html.Div("該当レースなし", style={"padding": "20px"})

    children = []
    for _, row in result.iterrows():
        children.append(search_race_button(row, row["レースid"]))
    return html.Div(children)


if __name__ == "__main__":
    app.run(debug=True)
