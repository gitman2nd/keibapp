from __future__ import annotations

from dash import Dash, html, dcc, Input, Output, State, ALL, no_update
import dash

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
# アプリ
# -------------------------
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server


def page_container():
    return html.Div(
        [
            dcc.Location(id="url"),
            dcc.Store(id="selected-race-id"),
            html.Div(id="page-content"),
        ],
        style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px"},
    )


# -------------------------
# 共通UI
# -------------------------
def race_button_card(race_row, button_id):
    text = f"{int(race_row['距離'])}m / {race_row['クラス']} / {int(race_row['出走頭数'])}頭"
    return html.Button(
        text,
        id=button_id,
        n_clicks=0,
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
    )


def search_race_button(race_row, button_id):
    text = f"{race_row['年']} / {race_row['月日']} / {int(race_row['距離'])}m / {race_row['クラス']}"
    return html.Button(
        text,
        id=button_id,
        n_clicks=0,
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
    )


def build_top_grid(latest_df):
    courses = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
    grid_rows = []

    header = [html.Div("", style={"fontWeight": "bold"})] + [
        html.Div(c, style={"fontWeight": "bold", "textAlign": "center"}) for c in courses
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
                grid_rows.append(
                    race_button_card(
                        race_row,
                        {"type": "race-link", "race_id": race_row["レースid"]},
                    )
                )

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
                                options=[{"label": str(x), "value": x} for x in FILTER_OPTIONS["years"]],
                                id="filter-year",
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("月"),
                            dcc.Dropdown(
                                options=[{"label": str(x), "value": x} for x in FILTER_OPTIONS["months"]],
                                id="filter-month",
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("芝砂"),
                            dcc.Dropdown(
                                options=[{"label": str(x), "value": x} for x in FILTER_OPTIONS["surfaces"]],
                                id="filter-surface",
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("距離"),
                            dcc.Dropdown(
                                options=[{"label": str(x), "value": x} for x in FILTER_OPTIONS["distances"]],
                                id="filter-distance",
                                multi=True,
                            ),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("クラス"),
                            dcc.Dropdown(
                                options=[{"label": str(x), "value": x} for x in FILTER_OPTIONS["classes"]],
                                id="filter-class",
                                multi=True,
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

    if not head:
        return html.Div(
            [
                dcc.Link("← TOPへ", href="/"),
                html.H2("レースが見つかりません"),
            ]
        )

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
                            "borderBottom": f"6px solid {row['waku_color']}",
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
            dcc.Link("← TOPへ", href="/"),
            html.H1(f"{head['コース名']} {int(head['R'])}R"),
            html.Div(
                [
                    html.Div(f"開催日: {head['開催日'].strftime('%Y-%m-%d') if head['開催日'] is not None else ''}"),
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
# レイアウト
# -------------------------
app.layout = page_container()


# -------------------------
# 画面切り替え
# -------------------------
@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname and pathname.startswith("/race/"):
        race_id = pathname.split("/race/")[-1]
        return build_race_page(race_id)
    return build_top_page()


# -------------------------
# 検索結果表示
# -------------------------
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
        return html.Div("該当レースなし")

    children = []
    for _, row in result.iterrows():
        children.append(
            search_race_button(
                row,
                {"type": "race-link", "race_id": row["レースid"]},
            )
        )
    return html.Div(children)


# -------------------------
# ボタン押下で遷移
# -------------------------
@app.callback(
    Output("url", "pathname"),
    Input({"type": "race-link", "race_id": ALL}, "n_clicks"),
    State({"type": "race-link", "race_id": ALL}, "id"),
    prevent_initial_call=True,
)
def move_race_page(n_clicks_list, ids):
    ctx = dash.callback_context
    if not ctx.triggered:
        return no_update

    triggered_id = ctx.triggered_id
    if not triggered_id:
        return no_update

    race_id = triggered_id.get("race_id")
    if not race_id:
        return no_update

    return f"/race/{race_id}"


if __name__ == "__main__":
    app.run(debug=True)