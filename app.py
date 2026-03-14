from dash import Dash, html, Input, Output, ALL, ctx
import pandas as pd

from data_loader import load_data
from analysis import make_race_plot_images

ur, ur_new, ur_pre = load_data()

btn_need_cols = ["コース_now", "R_now", "レースid_now"]
btn_exist_cols = [c for c in btn_need_cols if c in ur_pre.columns]

if len(btn_exist_cols) == 3:
    btn_df = (
        ur_pre[btn_need_cols]
        .dropna()
        .drop_duplicates()
        .sort_values(["コース_now", "R_now", "レースid_now"])
        .copy()
    )
    btn_df["label"] = btn_df["コース_now"].astype(str) + " × R" + btn_df["R_now"].astype(str)
else:
    btn_df = pd.DataFrame(columns=["コース_now", "R_now", "レースid_now", "label"])

def make_race_buttons(btn_df):
    buttons = []
    for i, row in btn_df.reset_index(drop=True).iterrows():
        buttons.append(
            html.Button(
                row["label"],
                id={"type": "race-btn", "index": i},
                n_clicks=0,
                style={"margin": "4px", "padding": "8px 12px", "cursor": "pointer"}
            )
        )
    return buttons

app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("競馬分析アプリ"),

    html.Hr(),

    html.H3("データ確認"),
    html.Div(f"ur_new shape: {ur_new.shape}"),
    html.Div(f"ur shape: {ur.shape}"),
    html.Div(f"ur_pre shape: {ur_pre.shape}"),

    html.Hr(),

    html.H3("レース選択"),
    html.Div(
        make_race_buttons(btn_df) if len(btn_df) else [html.Div("コース_now / R_now / レースid_now が見つかりません")],
        style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}
    ),

    html.Div(id="selected-race-text", style={"marginTop": "12px", "fontWeight": "bold"}),

    html.Hr(),

    html.H3("競馬グラフ"),
    html.Div(id="race-plot-area", children=[html.Div("ボタンを押してください")])
], style={"padding": "20px"})

@app.callback(
    Output("selected-race-text", "children"),
    Output("race-plot-area", "children"),
    Input({"type": "race-btn", "index": ALL}, "n_clicks")
)
def update_race_plots(n_clicks_list):
    if len(btn_df) == 0 or not n_clicks_list or max(n_clicks_list) == 0 or ctx.triggered_id is None:
        return "未選択", [html.Div("ボタンを押してください")]

    idx = ctx.triggered_id["index"]
    row = btn_df.reset_index(drop=True).iloc[idx]
    raceid_now = row["レースid_now"]
    label = row["label"]

    race_plot_components = make_race_plot_images(
        ur_pre=ur_pre,
        raceid_now=raceid_now,
        kaisai_now="2025-01-01",
        exclude=[]
    )
    return f"選択中: {label} / raceid_now={raceid_now}", race_plot_components

if __name__ == "__main__":
    app.run(debug=True)