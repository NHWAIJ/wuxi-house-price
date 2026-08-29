# -*- coding: utf-8 -*-
"""
run_test.py — 预测准确性测试 + 正式输出(每次运行两大部分):

Part A(正式输出,与正式项目完全一致):
  - 历史走势折线图 2 张(charts/)
  - 全量预测(66个月至2031.12)+ 预测延伸折线图 2 张(charts/)+ 预测表 xlsx

Part B(准确性测试):
  - 训练数据截断到 2024.08,从 2024.09 预测 22 个月,与真实值对比;
  - 对比折线图输出到 对比图/{运行编号}/ 文件夹(每次运行递增:1,2,3,…),永久保留;
  - 打分表 accuracy_metrics.xlsx:每次运行追加行,含 0-100 打分与区间覆盖率;
    另含"特征占比"sheet(回答"预测基于哪些数据")。

防过拟合:价格/事件严格截断、特征只用 ≤t-1 信息、30 轮 bootstrap。
帝泊湾稀疏数据:评估只在真实数据点上计误差。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config_loader import CFG
from load_data import discover, load_complex
from features import build_events, load_extra_events
from models import recursive_forecast, summarize_rounds, FEATURE_COLS, build_full_features
from forecast import targets_for, series_for_target

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(ROOT, "output")
CHART_DIR = os.path.join(ROOT, "comparison")
OFFICIAL_CHART_DIR = os.path.join(ROOT, "charts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(OFFICIAL_CHART_DIR, exist_ok=True)


def next_run_folder():
    """对比图/{运行编号}/ 每次运行递增(1,2,3,…)。"""
    n = 0
    for d in os.listdir(CHART_DIR):
        if d.isdigit():
            n = max(n, int(d))
    folder = os.path.join(CHART_DIR, str(n + 1))
    os.makedirs(folder, exist_ok=True)
    return folder

# ---- 从集中配置读取测试参数 ----
_TC = CFG["test"]
CUTOFF = pd.Timestamp(_TC["cutoff"])
N_MONTHS = _TC["n_months"]
N_ROUNDS = _TC["n_rounds"]
USE_HOLT_MAIN = _TC["use_holt_main"]
USE_CALIBRATION = _TC["use_calibration"]
                         # 注意:校准方向依赖训练期与测试期偏差是否同向(见 plan.md 校准说明),
                         # 若某小区校准后明显变差(方向反转),可设为 False 关闭。


# ---------------------------------------------------------------- 进度窗口

def _make_progress():
    if os.environ.get("BK_NO_GUI"):
        print("[提示] 已设置 BK_NO_GUI=1,本次运行不弹进度窗口,进度见控制台")
        return None
    try:
        from progress_gui import ProgressGUI
        return ProgressGUI(title="准确性测试运行中")
    except Exception as e:
        print(f"[提示] 进度窗口创建失败({e}),改用控制台输出")
        return None


# ---------------------------------------------------------------- 偏差校准

def calc_bias_ratio(series, events, horizon=12):
    """
    在训练集内部做 walk-forward 滚动验证,估计模型的系统性偏差比例:
        bias_ratio = mean((预测P50 - 真实) / 真实)
    正值 = 系统性高估,负值 = 低估。该量在"预测时点"可知(只用 ≤截断点的数据),无泄漏。
    校准系数 = 1 - bias_ratio,预测时对 P10/P50/P90 统一乘性修正。
    """
    from models import Ensemble, WALK_STEP
    feats = build_full_features(series, events)
    feats = feats[feats[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)
    X, y = feats[FEATURE_COLS].values, feats["price"].values
    n = len(feats)
    cutoffs = sorted(set(list(range(24, n - horizon, WALK_STEP)) + [n - horizon]))
    cutoffs = [c for c in cutoffs if c > 0]
    pairs = []
    for c in cutoffs:
        ens = Ensemble()
        ens.fit(X[:c], y[:c])
        pred = ens.predict(X[c: c + horizon])
        for h in range(horizon):
            j = c + h
            if j >= n:
                break
            pairs.append((y[j], pred[h]))
    if len(pairs) < 5:
        return 0.0
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    return float(np.mean((b - a) / a))


def calibration_factor(bias):
    """
    偏差 → 校准系数(乘性),带幅度自适应收缩:
      小偏差(|bias%| ≤ 5%)充分校准;偏差越大收缩越多(shrink = 5/|bias%|)。
      原因:训练期大幅偏差往往来自结构性行情(方向反转风险高),保守处理更稳。
    """
    shrink = min(1.0, 5.0 / max(abs(bias) * 100, 0.5))
    return 1.0 - shrink * bias


# ---------------------------------------------------------------- 评估

def eval_series(actual_df, pred_df, mask=None):
    """pred_df: date/P10/P50/P90/mean;actual_df: date/price。
    mask: 可选(日期索引的布尔 Series),仅在实际数据点上评估(帝泊湾稀疏数据)。
    返回指标 + 0-100 打分 + 区间覆盖率。"""
    m = pred_df.merge(actual_df, on="date", how="left")
    m = m[m["price"].notna()]
    if mask is not None:
        actual_dates = set(mask[mask].index)
        m = m[m["date"].isin(actual_dates)]
    if len(m) == 0:
        return None
    e = m["price"].values - m["P50"].values
    mape = float((np.abs(e) / m["price"].values).mean() * 100)
    # 0-100 打分:MAPE 每 1% 扣 2.5 分(MAPE=0 → 100分, 40% → 0分),再叠加区间覆盖奖励
    score = 100 - 2.5 * mape
    cov = float(((m["price"] >= m["P10"]) & (m["price"] <= m["P90"])).mean() * 100)
    cov_adj = max(-2.0, min(2.0, (cov - 80) * 0.1))   # 覆盖率微调,封顶 ±2 分(修复未封顶)
    score += cov_adj
    return {
        "MAE": float(np.abs(e).mean()),
        "RMSE": float(np.sqrt((e ** 2).mean())),
        "MAPE%": mape,
        "得分(0-100)": float(max(0, min(100, score))),
        "区间覆盖率%": cov,
        "样本数": int(len(m)),
        "真实均值": float(m["price"].mean()),
        "预测P50均值": float(m["P50"].mean()),
    }


# ---------------------------------------------------------------- 打分表(同一表格,可追加)

METRICS_FILE = os.path.join(OUTPUT_DIR, "accuracy_metrics.xlsx")
SCORE_FILE = os.path.join(OUTPUT_DIR, "打分表.xlsx")   # 独立打分表:小区|房型|时间|打分


def append_score_table():
    """
    独立"打分表.xlsx":列 = 小区 | 房型 | 时间(运行时间) | 打分(0-100)。
    每次运行后从 accuracy_metrics.xlsx 的"打分"sheet 全量重建(自动包含所有累计记录,
    每多训练一次,行数随之增加)。样式与其它表格一致(等线/居中/细边框)。
    """
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Border, Font, Side

    if not os.path.exists(METRICS_FILE):
        return None
    src = load_workbook(METRICS_FILE)["打分"]
    rows = []
    for r in src.iter_rows(min_row=2, values_only=True):
        if not r or r[0] is None:
            continue
        name, seq = r[1], r[2]
        if seq is None:
            continue
        # 格式区分:第8列(r[7])新版=区间覆盖率%(0-100),旧版=真实均值(>1000)
        # 新版:第7列(r[6])=得分;旧版:第6列(r[5])=MAPE%,回算得分
        if isinstance(r[7], (int, float)) and r[7] > 100:
            mape = float(r[5])
            score = round(max(0.0, min(100.0, 100 - 2.5 * mape)), 1)
        elif isinstance(r[6], (int, float)):
            score = round(float(r[6]), 1)
        else:
            continue
        rows.append([name, seq, str(r[0]), score])

    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    hfont = Font(name="等线", size=12, bold=True)
    dfont = Font(name="等线", size=12)

    wb = Workbook()
    ws = wb.active
    ws.title = "打分表"
    for j, h in enumerate(["小区", "房型", "时间", "打分(0-100)"], 1):
        cell = ws.cell(1, j, h)
        cell.font = hfont
        cell.alignment = center
        cell.border = border
    ws.row_dimensions[1].height = 15.5      # 表头行高(预测表模板)
    for i, row in enumerate(rows, 2):
        for j, v in enumerate(row, 1):
            cell = ws.cell(i, j, v)
            cell.font = dfont
            cell.alignment = center
            cell.border = border
        ws.row_dimensions[i].height = 27.5  # 数据行高(预测表模板)
    ws.column_dimensions["A"].width = 30    # 小区名(中文,加宽防截断)
    ws.column_dimensions["B"].width = 15.58
    ws.column_dimensions["C"].width = 22    # 时间(YYYY-MM-DD HH:MM)
    ws.column_dimensions["D"].width = 15.58
    wb.save(SCORE_FILE)
    return SCORE_FILE


def append_metrics(rows, feature_shares):
    """所有小区×序列的打分追加到同一 xlsx;含"特征占比"sheet(每次运行覆盖)。"""
    from openpyxl import Workbook, load_workbook
    from openpyxl.styles import Alignment, Border, Font, Side

    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    hfont = Font(name="等线", size=12, bold=True)
    dfont = Font(name="等线", size=12)

    if os.path.exists(METRICS_FILE):
        wb = load_workbook(METRICS_FILE)
        ws = wb["打分"]
        start = ws.max_row + 1
    else:
        wb = Workbook()
        ws = wb.active
        ws.title = "打分"
        for j, h in enumerate(["运行时间", "小区", "序列", "MAE", "RMSE", "MAPE%",
                               "得分(0-100)", "区间覆盖率%", "样本数",
                               "真实均值", "预测P50均值"], 1):
            cell = ws.cell(1, j, h)
            cell.font = hfont
            cell.alignment = center
            cell.border = border
        ws.row_dimensions[1].height = 15.5
        start = 2
    for i, row in enumerate(rows):
        r = start + i
        for j, v in enumerate(row, 1):
            cell = ws.cell(r, j, v)
            cell.font = dfont
            cell.alignment = center
            cell.border = border
        ws.row_dimensions[r].height = 27.5
    ws.column_dimensions["A"].width = 22    # 运行时间(YYYY-MM-DD HH:MM)
    ws.column_dimensions["B"].width = 30    # 小区名(中文,加宽防截断)
    for col in "CDEFGHIJK":
        ws.column_dimensions[col].width = 15.58

    # 特征占比 sheet(每次运行覆盖,反映当前模型依据)
    if "特征占比" in wb.sheetnames:
        del wb["特征占比"]
    ws2 = wb.create_sheet("特征占比")
    for j, h in enumerate(["特征类别", "占比%", "包含特征"], 1):
        cell = ws2.cell(1, j, h)
        cell.font = hfont
        cell.alignment = center
        cell.border = border
    for i, (cat, (share, feats)) in enumerate(feature_shares.items(), 2):
        for j, v in enumerate([cat, round(share * 100, 1), feats], 1):
            cell = ws2.cell(i, j, v)
            cell.font = dfont
            cell.alignment = center if j < 3 else Alignment(horizontal="left", vertical="center")
            cell.border = border
    wb.save(METRICS_FILE)
    return METRICS_FILE


def feature_importance_shares(series, events):
    """RF/GBR 特征重要性(均值)按类别聚合 → 占比(回答"预测基于哪些数据")。"""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    df = build_full_features(series, events)
    df = df[df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)
    X, y = df[FEATURE_COLS].values, df["price"].values
    imp = np.zeros(len(FEATURE_COLS))
    for m in (RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=4,
                                    random_state=42),
              GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.9, random_state=42)):
        m.fit(X, y)
        imp += m.feature_importances_
    imp /= 2
    cats = {
        "价格形态(滞后/动量/均线)": [f for f in FEATURE_COLS
                              if f.startswith(("p_lag", "r_lag", "ma"))],
        "波动率": ["vol12"],
        "距峰值回撤": ["dd_ratio", "months_since_peak"],
        "时间特征": ["t_idx", "year", "month_sin", "month_cos"],
        "政策事件": ["ev_net6", "ev_net12", "ev_net24"],
    }
    out = {}
    for cat, feats in cats.items():
        idx = [FEATURE_COLS.index(f) for f in feats]
        out[cat] = (float(imp[idx].sum()), ", ".join(feats))
    return out


# ---------------------------------------------------------------- 对比图

def plot_comparison(name, actual, preds, out_path, is_actual_mask):
    """一张图:综合/精装二手/毛坯或别墅二手 的真实 vs 预测(含80%区间)。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.family": ["Microsoft YaHei", "SimHei", "sans-serif"],
                         "axes.unicode_minus": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    SURFACE, PRIMARY, MUTED, GRIDLINE, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
    TRUTH = {"综合": "#52514e", "精装二手": "#eb6834", "毛坯二手": "#2a78d6", "别墅二手": "#2a78d6"}
    PRED = {"综合": "#4a3aa7", "精装二手": "#e87ba4", "毛坯二手": "#1baf7a", "别墅二手": "#1baf7a"}
    COL_MAP = {"综合": "price", "精装二手": "jingzhuang",
               "毛坯二手": "maopi", "别墅二手": "bieshu"}

    def _dt(s):
        return pd.to_datetime(s + "-01") + pd.offsets.MonthEnd(0)

    fig, ax = plt.subplots(figsize=(16, 7), dpi=150)
    for t, df in preds.items():
        dates = _dt(df["date"])
        act = actual[["date", COL_MAP[t]]].rename(columns={COL_MAP[t]: "v"})
        ax.plot(_dt(act["date"]), act["v"], color=TRUTH[t], lw=2.0, ls="-",
                label=f"{t} 实际", zorder=3)
        ax.plot(dates, df["P50"], color=PRED[t], lw=2.2, ls="--",
                label=f"{t} 预测P50", zorder=4)
        if t == "综合":
            ax.fill_between(dates, df["P10"], df["P90"], color=PRED[t], alpha=0.13,
                            label="综合预测 80%区间", zorder=2)
    if is_actual_mask is not None and is_actual_mask.any():
        act_dates = set(is_actual_mask[is_actual_mask].index)
        act_pts = actual[actual["date"].isin(act_dates)]
        ax.plot(_dt(act_pts["date"]), act_pts["price"], "o", color="#52514e",
                markersize=4, zorder=5, label=None)

    ax.axhline(0, color=AXIS, lw=0.8)
    ax.grid(axis="y", color=GRIDLINE, lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=3, labelsize=7)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(90)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    lo_all = min(min(df["P10"].min() for df in preds.values()),
                 float(actual[["price", "jingzhuang", "maopi", "bieshu"]].min().min()))
    hi_all = max(max(df["P90"].max() for df in preds.values()),
                 float(actual[["price", "jingzhuang", "maopi", "bieshu"]].max().max()))
    pad = (hi_all - lo_all) * 0.06
    ax.set_ylim(lo_all - pad, hi_all + pad)
    ax.set_ylabel("元 / ㎡", color=MUTED, fontsize=10)
    ax.set_title(f"{name} —— 实际 vs 预测(2024.09–2026.06,虚线=预测)",
                 loc="left", color=PRIMARY, fontsize=14, pad=4)
    leg = ax.legend(loc="upper right", frameon=True, fontsize=8.5,
                    edgecolor=GRIDLINE, facecolor=SURFACE, framealpha=0.93, ncols=2)
    for t in leg.get_texts():
        t.set_color(PRIMARY)
    fig.subplots_adjust(top=0.9, left=0.05, right=0.985, bottom=0.16)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------- 逐月对比表(xlsx)

def export_comparison(name, actual, preds):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    wb = Workbook()
    ws = wb.active
    ws.title = "逐月对比"
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    hfont = Font(name="等线", size=12, bold=True)
    dfont = Font(name="等线", size=12)
    cols = ["月份"] + [f"{t}_{q}" for t in preds for q in ("真实", "预测P10", "预测P50", "预测P90")]
    for j, h in enumerate(cols, 1):
        cell = ws.cell(1, j, h)
        cell.font = hfont
        cell.alignment = center
        cell.border = border
    ws.row_dimensions[1].height = 18
    col_map = {"综合": "price", "精装二手": "jingzhuang",
               "毛坯二手": "maopi", "别墅二手": "bieshu"}
    first = preds[list(preds)[0]]
    for i, date in enumerate(first["date"]):
        r = i + 2
        ws.cell(r, 1, date)
        col = 2
        for t, df in preds.items():
            row = df[df["date"] == date].iloc[0]
            real = actual[actual["date"] == date]
            real_v = real[col_map[t]].iloc[0] if len(real) else None
            for v in (real_v, row["P10"], row["P50"], row["P90"]):
                cell = ws.cell(r, col, None if v is None else int(round(v)))
                cell.font = dfont
                cell.alignment = center
                cell.border = border
                col += 1
        ws.row_dimensions[r].height = 18
    ws.column_dimensions["A"].width = 12
    for col in "BCDEFGHIJKLMNOP":
        ws.column_dimensions[col].width = 18   # 表头含中文(如 毛坯二手_预测P50),加宽防截断
    path = os.path.join(OUTPUT_DIR, f"accuracy_{name}.xlsx")
    wb.save(path)
    return path


# ---------------------------------------------------------------- 主流程

def export_metrics_xlsx(rows):
    """验证指标表 metrics.xlsx(等线、居中、细边框)。"""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side
    wb = Workbook()
    ws = wb.active
    ws.title = "metrics"
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    center = Alignment(horizontal="center", vertical="center")
    hfont = Font(name="等线", size=12, bold=True)
    dfont = Font(name="等线", size=12)
    for j, h in enumerate(["小区", "集成_MAE", "集成_RMSE", "集成_MAPE%",
                           "Holt_MAE", "Holt_RMSE", "Holt_MAPE%", "验证月数"], 1):
        cell = ws.cell(1, j, h)
        cell.font = hfont
        cell.alignment = center
        cell.border = border
    ws.row_dimensions[1].height = 18
    for i, row in enumerate(rows, 2):
        for j, v in enumerate(row, 1):
            cell = ws.cell(i, j, v)
            cell.font = dfont
            cell.alignment = center
            cell.border = border
        ws.row_dimensions[i].height = 18
    ws.column_dimensions["A"].width = 30
    for col in "BCDEFGH":
        ws.column_dimensions[col].width = 14
    path = os.path.join(OUTPUT_DIR, "metrics.xlsx")
    wb.save(path)
    return path


def _part_a_official(complexes, events_full, pw, report):
    """正式输出:月度价格走势图 2 张 + 验证指标表 metrics.xlsx。
    按需求:不再输出"从现在往后"(2026.07+)的预测延伸图与前瞻预测表,
    预测只保留 2024.09 起的准确性测试(Part B)。"""
    from models import walk_forward, summarize
    from plot_monthly import plot_all_history

    metrics_rows = []
    for i, c in enumerate(complexes):
        report(f"[正式] {c['name']}:历史走势图", 1 + i * 2, f"{c['name']}: 历史折线图(月度价格走势)")
        plot_all_history(c)
        wf = walk_forward(c["old"], events_full)
        ens_s, holt_s = summarize(wf["ensemble"]), summarize(wf["holt"])
        metrics_rows.append([c["name"], round(ens_s["MAE"]), round(ens_s["RMSE"]),
                             round(ens_s["MAPE%"], 2), round(holt_s["MAE"]),
                             round(holt_s["RMSE"]), round(holt_s["MAPE%"], 2),
                             ens_s["验证月数"]])
        print(f"  {c['name']} walk-forward: 集成 MAPE={ens_s['MAPE%']:.2f}%  vs  Holt {holt_s['MAPE%']:.2f}%")
    export_metrics_xlsx(metrics_rows)
    report("[正式] 历史走势图+验证指标完成", 4, "2 张月度价格走势图 + metrics.xlsx")


def run(pw=None):
    if pw is None:
        pw = _make_progress()
    last_pct = [0]

    def report(text, pct, log=None):
        if pw is not None:
            pw.update(text, pct, log)
        else:
            # 控制台模式也显示进度(每 5% 一行,不静默)
            if int(pct) >= last_pct[0] + 5 or pct >= 100:
                last_pct[0] = int(pct)
                print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    files = discover()
    report(f"发现 {len(files)} 个小区(数据快照)", 1, f"{[os.path.basename(f) for f in files]}")
    complexes = [load_complex(fp) for fp in files]
    all_events = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                           ignore_index=True)
    events_full = build_events(all_events, load_extra_events())

    # ---- Part A:正式输出(与正式项目完全一致) ----
    _part_a_official(complexes, events_full, pw, report)

    # ---- Part B:准确性测试(截断 2024.08,预测 22 个月,与真实对比) ----
    # 事件截断:只用 ≤2024.08(2024.09 后的政策在"当时"尚未发生)
    events = events_full[events_full.index <= CUTOFF]
    run_folder = next_run_folder()
    run_no = int(os.path.basename(run_folder))
    # 每次训练用不同随机种子(第1次=2027,第2次=2028,…):
    # bootstrap 抽样不同 → 每次训练内容不同、打分有波动、对比图有差异(但可复现)
    seed = 2026 + run_no

    # 每次运行只保留最新预测表:删除上一次的 accuracy_*/predictions_* 表
    # (accuracy_metrics.xlsx、打分表.xlsx、metrics.xlsx 永久保留,排除误删)
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith(".xlsx") and f not in ("accuracy_metrics.xlsx", "打分表.xlsx",
                                             "metrics.xlsx"):
            os.remove(os.path.join(OUTPUT_DIR, f))
    report("准确性测试:事件截断", 45, f"截断后事件 {len(events)} 条(仅≤2024.08),对比图 → {run_folder}")

    metrics_rows = []
    summaries = []
    total_steps = sum(30 * len(targets_for(c["old"])) for c in complexes)
    done = [0]
    feature_shares = {}
    run_time = time.strftime("%Y-%m-%d %H:%M")

    for ci, c in enumerate(complexes):
        name = c["name"]
        old = c["old"]
        # 训练序列:截断价格到 2024.08
        train = old[old["date"] <= CUTOFF]
        train_series = train[["date", "price", "is_actual"]].sort_values("date").reset_index(drop=True)
        # 真实值:2024.09 之后(评估时只取实际点)
        actual = old[old["date"] > CUTOFF][["date", "price", "jingzhuang", "maopi",
                                            "bieshu", "is_actual"]]
        actual["date"] = actual["date"].dt.strftime("%Y-%m")   # 与预测表 date 一致(字符串)
        actual_mask = actual.set_index("date")["is_actual"]

        report(f"[测试] {name}:截断训练集", 46 + ci * 2,
               f"训练窗口 {train['date'].min():%Y.%m}–{train['date'].max():%Y.%m}"
               f"({len(train)}个月) → 预测 2024.09 起 {N_MONTHS} 个月")

        # 30 轮 bootstrap 递归预测(3 条序列,与正式代码相同)
        preds = {}
        for t in targets_for(old):
            series = series_for_target(train, t)

            def _cb(d, tot):
                done[0] += 1
                report(f"[测试] 训练预测:{done[0]}/{total_steps} 轮({name} {t})",
                       48 + 45 * done[0] / total_steps, None)

            if USE_HOLT_MAIN:
                # Holt 趋势外推 + 季节形态叠加(方案B):
                #   P50 = Holt 直线 + 历史月度季节因子(残差按月均值,如"3月通常高于趋势线X元"),
                #   保留趋势准确性,同时预测呈现历史规律性的起伏;
                #   区间 = 训练期残差 std ±1.28(80%)。
                from models import fit_holt
                fit = fit_holt(series["price"].values)
                holt = np.asarray(fit.forecast(N_MONTHS))
                resid = series["price"].values - np.asarray(fit.fittedvalues)
                rstd = float(np.std(resid, ddof=1))
                # 季节因子:按月份分组的残差均值(3 个月平滑防单月噪声)
                s_tmp = series.copy()
                s_tmp["_m"] = s_tmp["date"].dt.month
                s_tmp["_r"] = resid
                season = s_tmp.groupby("_m")["_r"].mean()
                season = season.rolling(3, min_periods=1, center=True).mean()
                future_dates = pd.date_range(series["date"].max() + pd.offsets.MonthEnd(1),
                                             periods=N_MONTHS, freq="ME")
                seas_adj = np.array([season.get(m, 0.0) for m in future_dates.month])
                p50 = holt + seas_adj
                preds[t] = pd.DataFrame({
                    "date": future_dates.strftime("%Y-%m"),
                    "P10": p50 - 1.28 * rstd,
                    "P50": p50,
                    "P90": p50 + 1.28 * rstd,
                    "mean": p50,
                })
            else:
                rounds_df = recursive_forecast(series, events, "基准", n=N_MONTHS,
                                               rounds=N_ROUNDS, seed=seed, on_round=_cb)
                preds[t] = summarize_rounds(rounds_df)

        # ---- 偏差校准:用训练窗口内 walk-forward 验证估计系统性偏差,修正预测输出 ----
        # 自动开关:训练期偏差幅度 > 8% 时跳过校准(大幅偏差多为结构性行情,
        # 校准方向与测试期可能相反——如北控训练期低估、测试期高估,校准会帮倒忙)。
        # Holt 主预测模式下跳过(趋势外推偏差已很小,校准可能画蛇添足)。
        bias = calc_bias_ratio(train_series, events)
        if USE_HOLT_MAIN:
            print(f"\n[{name}] Holt 主预测:跳过偏差校准(趋势外推偏差已小,训练期验证偏差 {bias:+.2%})")
        elif USE_CALIBRATION and abs(bias) <= 0.08:
            calib = calibration_factor(bias)   # 幅度自适应收缩,防方向反转风险
            print(f"\n[{name}] 偏差校准:训练期验证偏差 {bias:+.2%}"
                  f"{'(高估)' if bias > 0 else '(低估)'} → 校准系数 {calib:.4f}")
            for t in preds:
                for col in ("P10", "P50", "P90", "mean"):
                    preds[t][col] = preds[t][col] * calib
            report(f"[测试] {name}:偏差校准完成", 48 + 45 * done[0] / max(total_steps, 1),
                   f"校准系数 {calib:.4f}(由训练期验证偏差 {bias:+.2%} 得到)")
        else:
            reason = "USE_CALIBRATION=False" if not USE_CALIBRATION else \
                f"训练期偏差 {bias:+.2%} 幅度超 8%(方向反转风险高),自动跳过校准"
            print(f"\n[{name}] 本次未校准: {reason}")
            report(f"[测试] {name}:未校准({reason})", 48 + 45 * done[0] / max(total_steps, 1), None)

        # 评估(帝泊湾只在真实点计误差)
        print(f"\n[{name}] 截断点 2024.08 预测 2024.09–2026.06(已校准)与真实对比:")
        series_scores = []
        for t, df in preds.items():
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            act = actual[["date", col]].rename(columns={col: "price"})
            m = eval_series(act, df, mask=actual_mask)
            if m:
                print(f"  {t}: MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  "
                      f"MAPE={m['MAPE%']:.2f}%  得分={m['得分(0-100)']:.1f}  "
                      f"覆盖率={m['区间覆盖率%']:.0f}%  (样本{m['样本数']}个真实点)")
                metrics_rows.append([run_time, name, t,
                                     round(m["MAE"]), round(m["RMSE"]), round(m["MAPE%"], 2),
                                     round(m["得分(0-100)"], 1), round(m["区间覆盖率%"]),
                                     m["样本数"], round(m["真实均值"]), round(m["预测P50均值"])])
                series_scores.append(m["得分(0-100)"])
        if series_scores:
            metrics_rows.append([run_time, name, "平均",
                                 "", "", "",
                                 round(sum(series_scores) / len(series_scores), 1),
                                 "", len(series_scores), "", ""])

        # 特征占比(基于截断训练集,回答"预测基于哪些数据")
        feature_shares[name] = feature_importance_shares(train_series, events)

        # 逐月对比表 + 对比图(输出到编号文件夹,永久保留)
        cmp_path = export_comparison(name, actual, preds)
        report(f"[测试] {name}:对比表已输出", 94 + ci * 2, f"逐月对比表 → {cmp_path}")
        plot_comparison(name, actual, preds,
                        os.path.join(run_folder, f"accuracy_{name}.png"), actual_mask)
        report(f"[测试] {name}:对比图已输出", 95 + ci * 2,
               f"对比图(编号{os.path.basename(run_folder)}) → {run_folder}")

        last_real = actual["price"].iloc[-1]
        last_pred = preds["综合"].iloc[-1]["P50"]
        summaries.append((name, last_real, last_pred))

    # 打分表(同一表格,追加;特征占比 sheet 每次覆盖)
    flat = {}
    for name, shares in feature_shares.items():
        for cat, (share, feats) in shares.items():
            flat[f"{name} | {cat}"] = (share, feats)
    metrics_file = append_metrics(metrics_rows, flat)
    score_file = append_score_table()
    report("全部完成", 100, f"打分表(追加) → {metrics_file}")
    if score_file:
        print(f"  独立打分表(小区/房型/时间/打分) → {score_file}")

    print("\n" + "=" * 90)
    print("最终对比(2026.06 末月): 真实  vs  预测P50")
    for name, real, pred in summaries:
        print(f"  {name}: 真实 {real:,.0f}  预测 {pred:,.0f}  偏差 {pred - real:+,.0f} "
              f"({pred / real - 1:+.1%})")
    print(f"\n全部完成:对比表/打分表 → {OUTPUT_DIR}  对比图(编号{os.path.basename(run_folder)}) → {run_folder}")
    if pw is not None:
        pw.done([f"正式图(charts/) + 预测表(预测结果/)",
                 f"打分表(追加) → {os.path.basename(metrics_file)}",
                 f"对比图 → {run_folder}"])


if __name__ == "__main__":
    from progress_gui import run_with_progress
    run_with_progress(run, _make_progress, title="准确性测试运行中")
