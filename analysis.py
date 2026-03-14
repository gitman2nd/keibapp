import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dash import html
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D

KYORI_MAP = {
    ("ダート", "短"): [1150, 1200, 1300, 1400, 1600, 1700],
    ("ダート", "中"): [1400, 1600, 1700, 1800, 1900, 2000, 2100],
    ("ダート", "長"): [1700, 1800, 1900, 2000, 2100],
    ("芝", "短"): [1200, 1400, 1500, 1600],
    ("芝", "中"): [1400, 1500, 1600, 1800, 2000],
    ("芝", "長"): [1600, 1800, 2000, 2200, 2400, 2500],
    ("芝", "超長"): [2200, 2400, 2500, 2600, 3000, 3200],
}

def log_fixed_slope(x, a, b):
    x = np.asarray(x, float)
    return a - 2.0 * np.log(x - b)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"

def make_race_plot_images(ur_pre, raceid_now, kaisai_now="2025-01-01", exclude=None):
    if exclude is None:
        exclude = []

    ziku, last, totyu = "コース", "上がり2f_a", "上がり2f以外_ab"
    RACE_ID_COL, UMA_COL, WAKU_COL, UMABAN_COL = "レースid_now", "馬名", "枠番", "馬番_now"

    kaisai_now = pd.Timestamp(kaisai_now)
    SELECT_UMABAN = [i for i in range(1, 19) if i not in exclude]

    raceid_now = str(raceid_now)

    target_rows = ur_pre.loc[ur_pre[RACE_ID_COL].astype(str) == raceid_now].copy()
    if len(target_rows) == 0:
        return [html.Div(f"raceid_now={raceid_now} が見つかりません")]

    base_row = target_rows.iloc[0]

    rank_col = "クラス_now" if "クラス_now" in ur_pre.columns else "クラス"
    shiba_col = "芝砂_now" if "芝砂_now" in ur_pre.columns else "芝砂"
    kyori_kubun_col = "距離区分_now" if "距離区分_now" in ur_pre.columns else "距離区分"

    Rank = base_row[rank_col]
    SHIBASUNA = base_row[shiba_col]
    KYORI_KUBUN = base_row[kyori_kubun_col]
    KYORI_LIST = KYORI_MAP.get((SHIBASUNA, KYORI_KUBUN), [])

    if len(KYORI_LIST) == 0:
        return [html.Div(f"距離リストが見つかりません: {(SHIBASUNA, KYORI_KUBUN)}")]

    result_components = []

    for KYORI in KYORI_LIST:
        if isinstance(Rank, (list, tuple, set, pd.Series, np.ndarray)):
            class_mask = ur_pre["クラス"].isin(Rank)
        else:
            class_mask = ur_pre["クラス"].eq(Rank)

        df = ur_pre.loc[
            (ur_pre["芝砂"] == SHIBASUNA)
            & class_mask
            & (pd.to_numeric(ur_pre["タイムz"], errors="coerce") >= 30)
            & (pd.to_numeric(ur_pre["距離"], errors="coerce") == KYORI)
        ].copy()

        df = df.dropna(subset=[totyu, last, ziku, "開催日", UMA_COL])

        df_now = ur_pre.loc[
            (ur_pre[RACE_ID_COL].astype(str) == raceid_now)
            & (pd.to_numeric(ur_pre["距離"], errors="coerce") == KYORI)
            & (ur_pre["芝砂"] == SHIBASUNA)
            & (ur_pre["開催日"] > kaisai_now)
        ].copy()

        if SELECT_UMABAN is not None and UMABAN_COL in df_now.columns:
            df_now = df_now[df_now[UMABAN_COL].isin(SELECT_UMABAN)]

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = plt.cm.tab10.colors
        course_color = {}

        source_courses = pd.unique(df_now[ziku]) if len(df_now) else pd.unique(df[ziku])

        for i, cls in enumerate(sorted(source_courses)):
            c = colors[i % len(colors)]
            course_color[cls] = c
            course_color[str(cls)] = c

            sub = df[df[ziku] == cls]
            ax.scatter(sub[totyu], sub[last], alpha=0.65, color=c, s=25)

            x = pd.to_numeric(sub[totyu], errors="coerce").to_numpy(float)
            y = pd.to_numeric(sub[last], errors="coerce").to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            x, y = x[m], y[m]

            if x.size < 5:
                continue

            b0 = x.min() - 1e-3
            a0 = float(np.mean(y + 2.0 * np.log(x - b0)))

            try:
                popt, _ = curve_fit(
                    log_fixed_slope,
                    x,
                    y,
                    p0=[a0, b0],
                    bounds=([-np.inf, -np.inf], [np.inf, x.min() - 1e-6]),
                    maxfev=20000
                )
                a_hat, b_hat = map(float, popt)
                xx = np.linspace(x.min(), x.max(), 200)
                ax.plot(xx, log_fixed_slope(xx, a_hat, b_hat), color=c, lw=2, ls="--", label=str(cls))
            except Exception:
                pass

        now_labels, items = [], []

        if len(df_now):
            need_cols = [totyu, last, ziku, UMA_COL, WAKU_COL, UMABAN_COL, "テン1F"]
            lack_cols = [c for c in need_cols if c not in df_now.columns]

            if len(lack_cols) == 0:
                df_now = df_now.dropna(subset=need_cols).copy()
                waku_fill = {1: "white", 2: "black", 3: "red", 4: "blue", 5: "yellow", 6: "lime", 7: "orange", 8: "magenta"}
                markers = ["o", "s", "D"]

                for w, gg in df_now.groupby(WAKU_COL):
                    gg = gg.sort_values(UMABAN_COL)
                    for j, u in enumerate(pd.unique(gg[UMA_COL])):
                        g = gg[gg[UMA_COL] == u]
                        fill = waku_fill.get(int(w), "gray")
                        mk = markers[j % len(markers)]
                        umaban = int(g[UMABAN_COL].iloc[0])
                        cols = [course_color.get(c, course_color.get(str(c), "black")) for c in g[ziku].tolist()]

                        ax.scatter(
                            g[totyu], g[last],
                            s=150, color=cols, marker=mk,
                            edgecolor=fill, linewidth=2.6, alpha=1.0, zorder=10
                        )

                        for xi, yi, co in zip(
                            pd.to_numeric(g[totyu], errors="coerce").to_numpy(float),
                            pd.to_numeric(g[last], errors="coerce").to_numpy(float),
                            g["テン1F"].astype(str).to_numpy()
                        ):
                            if np.isfinite(xi) and np.isfinite(yi):
                                ax.text(xi, yi, str(co), color="black", fontsize=6, fontweight="bold",
                                        ha="center", va="center", zorder=11)

                        t1 = ",".join(map(str, sorted(pd.unique(g["テン1F"].astype(str)))))
                        items.append((umaban, f"{umaban}:{u}({t1})", fill, mk))

                items.sort(key=lambda t: t[0])
                now_labels = [t[1] for t in items]
                now_leg_handles = [
                    Line2D([0], [0], marker=t[3], linestyle="None",
                           markerfacecolor=("black" if t[2] == "white" else "white"),
                           markeredgecolor=t[2], markeredgewidth=2.0, markersize=10)
                    for t in items
                ]

                h1, l1 = ax.get_legend_handles_labels()
                if len(l1):
                    leg1 = ax.legend(h1, l1, fontsize=8, loc="upper left",
                                     bbox_to_anchor=(1.02, 1.0), borderaxespad=0.,
                                     title="FIT", title_fontsize=8)
                    ax.add_artist(leg1)

                if len(now_labels):
                    ax.legend(now_leg_handles, now_labels, fontsize=8, loc="lower left",
                              bbox_to_anchor=(1.02, 0.0), borderaxespad=0.,
                              title="NOW", title_fontsize=8, markerscale=0.6, frameon=False)

        ax.set_xlabel("道中平均（上がり2f以外）")
        ax.set_ylabel("ラスト2F平均（上がり2f）")
        ax.set_title(f"{SHIBASUNA} 距離={KYORI}")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, which="major", color="white", linewidth=1.0)
        fig.tight_layout(rect=[0, 0.25, 1, 1])

        img_src = fig_to_base64(fig)
        result_components.append(
            html.Div([
                html.H4(f"{SHIBASUNA} 距離={KYORI}"),
                html.Img(src=img_src, style={"width": "100%", "maxWidth": "1200px"})
            ], style={"marginBottom": "40px"})
        )

    return result_components