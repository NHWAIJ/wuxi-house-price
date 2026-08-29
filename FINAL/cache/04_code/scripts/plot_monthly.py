# -*- coding: utf-8 -*-
"""
plot_monthly.py — 历史月度价格走势图(每小区一张,四线版,可独立运行)。

历史四线:毛坯/精装/别墅 × 新房(实线)/二手房(虚线),类型按表内可用数据自适应。
X 轴每 3 个月标注"年-月",图拉长,附关键事件标注与布局重叠检查。
由 run_all.py 一键调用(与预测延伸图一并输出),也可单独运行:
  python scripts/plot_monthly.py
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
from load_data import discover, load_complex
from plot_forecast import _history_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, "charts")
os.makedirs(CHART_DIR, exist_ok=True)

SURFACE = "#fcfcfb"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
    "axes.unicode_minus": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

EVENT_SPECS = {
    "北控雁栖湖_价格数据表": [
        ("2019-09", "首开毛坯 13,000\n三开三罄", True, 13000),
        ("2020-08", "8·30新政(调控升级)", True, 17000),
        ("2021-03", "二手房入市", False, 15813),
        ("2021-07", "二手房成交参考价机制\n(江苏首个)", True, 17014),
        ("2024-01", "精装峰值 17,182", False, 17182),
        ("2026-06", "谷值 毛坯13,097\n精装13,490", False, 13490),
    ],
    "新力帝泊湾_价格数据表": [
        ("2018-12", "首开精装 13,000", True, 13000),
        ("2020-06", "全部售罄\n转入二手房", True, 14800),
        ("2022-07", "精装跌破万元\n9,881", False, 9881),
        ("2024-07", "新低 精装 8,806", True, 8806),
        ("2026-06", "历史最低\n精装 8,535", False, 8535),
    ],
}


def draw(name, complex_dict):
    new, old = complex_dict["new"], complex_dict["old"]
    fig, ax = plt.subplots(figsize=(24, 7.6), dpi=150)

    handles = _history_lines(complex_dict, ax)

    # y 范围:历史数据自适应,留 6% 余量
    price_cols = [c for c in ("maopi", "jingzhuang", "bieshu", "zonghe")
                  if c in old.columns and old[c].notna().any()]
    all_vals = pd.concat([old[c] for c in price_cols] +
                         [new[c] for c in price_cols if c in new.columns and new[c].notna().any()])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    pad = (hi - lo) * 0.07
    ax.set_ylim(lo - pad, hi + pad)

    # 轴与网格
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
    ax.set_ylabel("元 / ㎡", color=MUTED, fontsize=10, labelpad=6)

    # 标题与副标题
    ax.set_title(f"{name} —— 月度价格走势(开盘 → {old['date'].max():%Y.%m})",
                 loc="left", color=PRIMARY, fontsize=15, pad=4)
    fig.text(0.045, 0.90,
             "实线=新房(售罄段) · 虚线=二手房 · 单位:元/㎡ · "
             "数据来源见各表(二手房毛坯/精装为市价差估算;帝泊湾二手为季度采样插值,圆点为实际数据)",
             color=MUTED, fontsize=8.4)

    # 事件标注
    for date, label, above, y_at in EVENT_SPECS.get(name, []):
        d = pd.Timestamp(date + "-01") + pd.offsets.MonthEnd(0)
        gap = (hi - lo) * 0.02
        y_text = y_at + gap if above else y_at - gap
        ax.annotate(label, xy=(d, y_at), xytext=(d, y_text), textcoords="data",
                    ha="center", va="bottom" if above else "top",
                    fontsize=8.5, color=SECONDARY, linespacing=1.35,
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8,
                                    shrinkA=0, shrinkB=2, mutation_scale=6))

    # 图例(右上,白底)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=9.5,
                    edgecolor=GRIDLINE, facecolor=SURFACE, framealpha=0.92,
                    handlelength=3.0, ncols=2)
    for t in leg.get_texts():
        t.set_color(PRIMARY)

    fig.subplots_adjust(top=0.87, left=0.05, right=0.985)

    # 布局检查
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

    out = os.path.join(CHART_DIR, f"{name}_月度价格走势.png")
    fig.savefig(out)
    plt.close(fig)
    print("已生成:", out)


def plot_all_history(complex_dict):
    draw(complex_dict["name"], complex_dict)


if __name__ == "__main__":
    for fp in discover():
        plot_all_history(load_complex(fp))
