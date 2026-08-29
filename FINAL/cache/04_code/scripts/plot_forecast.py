# -*- coding: utf-8 -*-
"""
plot_forecast.py — 预测延伸折线图(代码独立完成训练与出图,输出到 charts/)。

历史段:各装修类型新房(实线)/二手房(虚线) + 二手房综合均价(深灰细线);
预测段(2026.07 → 2031.12,每月):
  综合均价   基准 P50(紫色粗实线) + 预测区间带(P10-P90,淡紫,百分比随配置);
  精装二手   基准 P50(品红);
  毛坯二手/别墅二手(按表内可用类型)基准 P50(青色)。
X 轴精确到每月:逐月刻度 + 逐月标签(竖排),图整体拉长。
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forecast import targets_for, series_for_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"          # 历史:毛坯/别墅
ORANGE = "#eb6834"        # 历史:精装
HIST_ZONGHE = "#52514e"   # 历史:二手房综合均价
PRED_COLOR = {            # 预测段颜色(与历史蓝/橙/灰区分)
    "综合": "#4a3aa7",        # 紫
    "精装二手": "#e87ba4",    # 品红
    "毛坯二手": "#1baf7a",    # 青
    "别墅二手": "#1baf7a",    # 青
}
BAND_ALPHA = 0.15
TYPE_LABEL = {"maopi": "毛坯", "jingzhuang": "精装", "bieshu": "别墅"}

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

EVENT_SPECS = {
    "北控雁栖湖_价格数据表": [
        ("2019-09", "首开毛坯 13,000", True, 13000),
        ("2021-07", "二手房参考价机制", True, 17014),
        ("2024-01", "精装峰值 17,182", False, 17182),
    ],
    "新力帝泊湾_价格数据表": [
        ("2018-12", "首开精装 13,000", True, 13000),
        ("2022-07", "精装跌破万元", False, 9881),
        ("2024-07", "新低 精装 8,806", True, 8806),
    ],
}


def _interval_pct():
    """当前区间百分比(延迟 import 防循环依赖)。"""
    from predict import INTERVAL_QUANTILES
    return (INTERVAL_QUANTILES[1] - INTERVAL_QUANTILES[0],)


def _history_lines(complex_dict, ax):
    """自适应画历史四线(新房实线/二手虚线),返回图例句柄。"""
    new, old = complex_dict["new"], complex_dict["old"]
    handles = []
    for t in ("maopi", "jingzhuang", "bieshu"):
        has_new = t in new.columns and new[t].notna().any()
        has_old = t in old.columns and old[t].notna().any()
        if not (has_new or has_old):
            continue
        color = BLUE if t != "jingzhuang" else ORANGE
        if has_new:
            h, = ax.plot(new["date"], new[t], color=color, lw=2.0, ls="-",
                         solid_joinstyle="round", zorder=3, label=f"{TYPE_LABEL[t]}新房")
            handles.append(h)
        if has_old:
            h, = ax.plot(old["date"], old[t], color=color, lw=2.0, ls="--",
                         solid_joinstyle="round", zorder=3, label=f"{TYPE_LABEL[t]}二手")
            handles.append(h)
    return handles


def draw(name, complex_dict, result, out_path=None):
    """out_path: 指定输出路径(默认 charts/);用于与对比图放同一文件夹。"""
    old = complex_dict["old"]
    fig, ax = plt.subplots(figsize=(34, 7.6), dpi=150)

    # ---- 历史段 ----
    handles = _history_lines(complex_dict, ax)
    h_z, = ax.plot(old["date"], old["zonghe"], color=HIST_ZONGHE, lw=1.5, ls="-",
                   zorder=2, label="二手房综合均价")
    handles.append(h_z)

    # ---- 预测段:综合(主)+ 细分序列,从各自历史末端延伸 ----
    pred_handles = []
    for t in targets_for(old):
        hist_series = series_for_target(old, t)
        last_price = hist_series["price"].iloc[-1]
        df = result[t]
        dates = pd.to_datetime(df["date"] + "-01") + pd.offsets.MonthEnd(0)
        dts = np.concatenate([[hist_series["date"].max()], dates])
        if t == "综合":
            mid = np.concatenate([[last_price], df["P50"].values])
            lo = np.concatenate([[last_price], df["P10"].values])
            hi = np.concatenate([[last_price], df["P90"].values])
            h, = ax.plot(dts, mid, color=PRED_COLOR[t], lw=2.6, ls="-",
                         solid_joinstyle="round", zorder=5,
                         label=f"综合均价预测(基准P50)")
            ax.fill_between(dts, lo, hi, color=PRED_COLOR[t], alpha=BAND_ALPHA,
                            zorder=4, label="基准 {}-{}%区间(P10-P90)".format(*_interval_pct()))
        else:
            h, = ax.plot(dts, np.concatenate([[last_price], df["P50"].values]),
                         color=PRED_COLOR[t], lw=2.0, ls="--",
                         solid_joinstyle="round", zorder=5,
                         label=f"{t}预测(P50)")
        pred_handles.append(h)
    # 预测起点竖线
    start_date = old["date"].max()
    ax.axvline(start_date, color=AXIS, lw=0.8, ls=":", zorder=1)

    # ---- 轴与网格 ----
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="y", color=GRIDLINE, lw=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=MUTED, length=3, labelsize=6)
    ax.tick_params(axis="y", colors=MUTED, length=3, labelsize=9)
    # X 轴精确到每月:逐月刻度 + 逐月标签(竖排)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
        lbl.set_ha("center")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # y 范围:历史+预测整体 min/max 留余量
    hist_vals = old[["zonghe", "maopi", "jingzhuang", "bieshu"]].min().min()
    all_hi = max(old[["zonghe", "maopi", "jingzhuang", "bieshu"]].max().max(),
                 max(result[t]["P90"].max() for t in targets_for(old)))
    all_lo = min(hist_vals, min(result[t]["P10"].min() for t in targets_for(old)))
    pad = (all_hi - all_lo) * 0.07
    ax.set_ylim(all_lo - pad, all_hi + pad)
    ax.set_ylabel("元 / ㎡", color=MUTED, fontsize=10, labelpad=6)

    # ---- 标题 ----
    ax.set_title(f"{name} —— 月度价格走势与基准预测(至2031.12,每月)",
                 loc="left", color=PRIMARY, fontsize=15, pad=4)
    fig.text(0.03, 0.90,
             f"历史段:开盘→{old['date'].max():%Y.%m} · 预测段:基准情景,延续近期政策事件节奏 · "
             f"紫线=综合均价基准预测,色带={_interval_pct()[0]}%区间 · 品红=精装二手预测 · 青=毛坯/别墅二手预测",
             color=MUTED, fontsize=8.4)

    # ---- 事件标注(历史段) ----
    for date, label, above, y_at in EVENT_SPECS.get(name, []):
        d = pd.Timestamp(date + "-01") + pd.offsets.MonthEnd(0)
        y_lim = ax.get_ylim()
        gap = (y_lim[1] - y_lim[0]) * 0.02
        ax.annotate(label, xy=(d, y_at), xytext=(d, y_at + gap if above else y_at - gap),
                    textcoords="data", ha="center", va="bottom" if above else "top",
                    fontsize=8.5, color=SECONDARY, linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8,
                                    shrinkA=0, shrinkB=2, mutation_scale=6))

    # ---- 图例(右上,两列) ----
    leg = ax.legend(loc="upper right", frameon=True, fontsize=9,
                    edgecolor=GRIDLINE, facecolor=SURFACE, framealpha=0.93,
                    handlelength=2.6, ncols=2)
    for t in leg.get_texts():
        t.set_color(PRIMARY)

    fig.subplots_adjust(top=0.87, left=0.035, right=0.985)

    # ---- 布局检查 ----
    fig.canvas.draw()
    r = fig.canvas.get_renderer()

    def overlap(a, b):
        ix = min(a.x1, b.x1) - max(a.x0, b.x0)
        iy = min(a.y1, b.y1) - max(a.y0, b.y0)
        return ix > 1 and iy > 1

    xt = [t.get_window_extent(r) for t in ax.get_xticklabels()]
    issues = []
    if any(overlap(xt[i], xt[i + 1]) for i in range(len(xt) - 1)):
        issues.append("X 轴刻度标签重叠")
    boxes = [(t.get_window_extent(r), t.get_text().replace("\n", " / ")[:16]) for t in ax.texts]
    leg_bbox = leg.get_window_extent(r)
    for i in range(len(boxes)):
        if overlap(boxes[i][0], leg_bbox):
            issues.append(f"标注[{boxes[i][1]}] 与图例重叠")
        for j in range(i + 1, len(boxes)):
            if overlap(boxes[i][0], boxes[j][0]):
                issues.append(f"标注[{boxes[i][1]}] 与 [{boxes[j][1]}] 重叠")
    print(f"[{name}] {'⚠ ' + str(issues) if issues else '✓ 无布局问题'}")

    out = out_path or os.path.join(CHART_DIR, f"{name}_预测延伸.png")
    fig.savefig(out)
    plt.close(fig)
    print("已生成:", out)


def plot_all(complex_dict, result):
    draw(complex_dict["name"], complex_dict, result)
