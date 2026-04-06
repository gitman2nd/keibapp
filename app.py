from __future__ import annotations

from dash import Dash, html, dcc, Input, Output, State, ALL, no_update
import logging

from data_loader import load_race_data
from analysis import (
    build_race_master,
    get_latest_week_races,
    filter_races,
    get_filter_options,
    get_race_detail,
)

DF = load_race_data("data/ur_bunseki.parquet")
RACE_MASTER = build_race_master(DF)
FILTER_OPTIONS = get_filter_options(RACE_MASTER)

logging.basicConfig(
    filename="dash_app.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def race_button_card(race_row, race_id):
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
            html.Div(
                [
                    html.H1("野球アプリ", style={"margin": "0"}),
                    dcc.Link(
                        html.Button(
                            "AI予測",
                            style={
                                "padding": "10px 24px",
                                "backgroundColor": "#1976d2",
                                "color": "#ffffff",
                                "border": "none",
                                "borderRadius": "8px",
                                "fontSize": "16px",
                                "fontWeight": "bold",
                                "cursor": "pointer",
                            },
                        ),
                        href="/predict",
                    ),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "marginBottom": "24px",
                },
            ),
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


def build_predict_top_page():
    return html.Div(
        [
            dcc.Link(
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
            ),
            html.H1("AI予測 - レース選択"),
            html.P("予測したいレースを選択してください"),
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
            html.Div(id="predict-search-result-area"),
        ]
    )


def build_predict_result_page(race_id: str, model_version: str = "v1"):
    from predict import predict_race_with_details, get_model_info
    import pandas as pd

    model_info = get_model_info(model_version)
    head, result = predict_race_with_details(DF, race_id, model_version)

    back_link = dcc.Link(
        "← 予測TOPへ",
        href="/predict",
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

    if result.empty:
        return html.Div(
            [
                dcc.Store(id="selected-model-store", data=model_version),
                back_link,
                html.H2("予測モデルまたはデータが見つかりません"),
            ]
        )

    def get_table_row(rank, row, is_top3=False):
        bg_color = "#fff9e6" if is_top3 else "#ffffff"
        border_color = "#ffd700" if is_top3 else "#e0e0e0"

        odds_value = row.get("単オッズ")
        odds_text = f"オッズ {odds_value:.1f}" if pd.notna(odds_value) else ""

        return html.Div(
            [
                html.Div(
                    f"{rank}",
                    style={
                        "width": "40px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "fontSize": "18px",
                    },
                ),
                html.Div(
                    row["馬名"],
                    style={
                        "flex": 2,
                        "fontWeight": "500",
                        "fontSize": "15px",
                    },
                ),
                html.Div(
                    f"{row['予測確率']}%",
                    style={
                        "width": "80px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "fontSize": "18px",
                        "color": "#1976d2",
                    },
                ),
                html.Div(
                    odds_text,
                    style={"width": "100px", "color": "#666", "fontSize": "13px"},
                ),
                html.Div(
                    row.get("騎手", ""),
                    style={"flex": 1, "color": "#666", "fontSize": "13px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "padding": "12px",
                "background": bg_color,
                "border": f"2px solid {border_color}",
                "borderRadius": "8px",
                "marginBottom": "6px",
            },
        )

    model_selector = html.Div(
        [
            html.Label(
                "モデル切替:", style={"marginRight": "10px", "fontWeight": "bold"}
            ),
            dcc.RadioItems(
                id="model-selector",
                options=[
                    {"label": "v1 (単オディズ含む)", "value": "v1"},
                    {"label": "v2 (単オディズ除外)", "value": "v2"},
                ],
                value=model_version,
                inline=True,
                style={"display": "inline-block"},
            ),
            html.Div(
                f"特徴量: {', '.join(model_info['feature_cols'])}",
                style={"fontSize": "12px", "color": "#666", "marginTop": "8px"},
            ),
        ],
        style={
            "marginBottom": "20px",
            "padding": "16px",
            "background": "#f5f5f5",
            "borderRadius": "8px",
        },
    )

    race_info = [
        html.Div(
            [
                html.Div(
                    f"{head['コース名']} {int(head['R'])}R",
                    style={"fontSize": "24px", "fontWeight": "bold"},
                ),
                html.Div(
                    f"{head['開催日'].strftime('%Y-%m-%d') if pd.notna(head.get('開催日')) else ''}",
                    style={"color": "#666"},
                ),
            ]
        ),
        html.Div(
            [
                html.Div(f"距離: {int(head['距離'])}m"),
                html.Div(f"クラス: {head['クラス']}"),
                html.Div(f"芝砂: {head['芝砂']}"),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(3, 1fr)",
                "gap": "8px",
                "background": "#f5f5f5",
                "padding": "16px",
                "borderRadius": "8px",
            },
        ),
    ]

    horses = []
    for idx, (_, row) in enumerate(result.iterrows()):
        is_top3 = idx < 3
        horses.append(get_table_row(idx + 1, row, is_top3))

    return html.Div(
        [
            dcc.Store(id="selected-model-store", data=model_version),
            dcc.Store(id="race-id-store", data=race_id),
            back_link,
            html.H1("AI予測結果"),
            model_selector,
            html.Div(
                race_info,
                style={"marginBottom": "24px"},
            ),
            html.Div(
                html.Div(
                    [
                        html.Div(
                            "予測順位", style={"width": "40px", "fontWeight": "bold"}
                        ),
                        html.Div("馬名", style={"flex": 2, "fontWeight": "bold"}),
                        html.Div(
                            "予測確率",
                            style={
                                "width": "80px",
                                "fontWeight": "bold",
                                "textAlign": "center",
                            },
                        ),
                        html.Div(
                            "オッズ", style={"width": "100px", "fontWeight": "bold"}
                        ),
                        html.Div("騎手", style={"flex": 1, "fontWeight": "bold"}),
                    ],
                    style={
                        "display": "flex",
                        "padding": "8px 12px",
                        "background": "#f0f0f0",
                        "borderRadius": "8px",
                        "marginBottom": "8px",
                    },
                )
            ),
            html.Div(id="prediction-results", children=horses),
            html.Div(
                "※ 本予測は機械学習モデルによる参考値です。実際の投注は自己責任で行ってください。",
                style={
                    "marginTop": "24px",
                    "color": "#888",
                    "fontSize": "12px",
                    "textAlign": "center",
                },
            ),
        ]
    )


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    logging.info(f"アクセスされたURL: {pathname}")
    if pathname and pathname.startswith("/race/"):
        race_id = pathname.split("/race/")[-1]
        return build_race_page(race_id)
    if pathname and pathname.startswith("/predict/"):
        race_id = pathname.split("/predict/")[-1]
        return build_predict_result_page(race_id)
    if pathname == "/predict":
        return build_predict_top_page()
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


@app.callback(
    Output("predict-search-result-area", "children"),
    Input("filter-year", "value"),
    Input("filter-month", "value"),
    Input("filter-surface", "value"),
    Input("filter-distance", "value"),
    Input("filter-class", "value"),
)
def update_predict_search_results(years, months, surfaces, distances, classes):
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
        children.append(
            dcc.Link(
                html.Button(
                    f"{row['年']} / {row['月日']} / {int(row['距離'])}m / {row['クラス']}",
                    style={
                        "width": "100%",
                        "padding": "10px 12px",
                        "marginBottom": "8px",
                        "borderRadius": "10px",
                        "border": f"2px solid {row['surface_color']}",
                        "background": "#ffffff",
                        "cursor": "pointer",
                        "textAlign": "left",
                    },
                ),
                href=f"/predict/{row['レースid']}",
            )
        )
    return html.Div(children)


@app.callback(
    Output("prediction-results", "children"),
    Input("model-selector", "value"),
    Input("race-id-store", "data"),
)
def update_prediction_results(model_version, race_id):
    from predict import predict_race_with_details
    import pandas as pd

    _, result = predict_race_with_details(DF, race_id, model_version)

    def get_table_row(rank, row, is_top3=False):
        bg_color = "#fff9e6" if is_top3 else "#ffffff"
        border_color = "#ffd700" if is_top3 else "#e0e0e0"

        odds_value = row.get("単オッズ")
        odds_text = f"オッズ {odds_value:.1f}" if pd.notna(odds_value) else ""

        return html.Div(
            [
                html.Div(
                    f"{rank}",
                    style={
                        "width": "40px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "fontSize": "18px",
                    },
                ),
                html.Div(
                    row["馬名"],
                    style={
                        "flex": 2,
                        "fontWeight": "500",
                        "fontSize": "15px",
                    },
                ),
                html.Div(
                    f"{row['予測確率']}%",
                    style={
                        "width": "80px",
                        "fontWeight": "bold",
                        "textAlign": "center",
                        "fontSize": "18px",
                        "color": "#1976d2",
                    },
                ),
                html.Div(
                    odds_text,
                    style={"width": "100px", "color": "#666", "fontSize": "13px"},
                ),
                html.Div(
                    row.get("騎手", ""),
                    style={"flex": 1, "color": "#666", "fontSize": "13px"},
                ),
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "gap": "12px",
                "padding": "12px",
                "background": bg_color,
                "border": f"2px solid {border_color}",
                "borderRadius": "8px",
                "marginBottom": "6px",
            },
        )

    horses = []
    for idx, (_, row) in enumerate(result.iterrows()):
        is_top3 = idx < 3
        horses.append(get_table_row(idx + 1, row, is_top3))

    return horses


if __name__ == "__main__":
    app.run(debug=True)
