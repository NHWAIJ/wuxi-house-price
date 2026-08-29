# -*- coding: utf-8 -*-
"""
target_diagnose.py — 目标小区诊断模式(对应工作流:截断→预测→打分→调整→正式预测):
  1. 对目标小区在多个截断点(2022.03 / 2023.04 / 2024.05 / 2025.06 等,自动选取)
     各做一次"截断→Holt 预测 12 个月→与真实对比打分"(只用截断前数据,严格无泄漏);
  2. 汇总诊断表:各窗口 MAE/得分/偏差 → 判断该小区 Holt 预测的稳定性与置信度
     (多次窗口一致=高置信;分数差异大=低置信,正式预测需谨慎解读);
  3. 自动进入正式预测:全部历史 → 2031.12 → 预测表(图由 predict 流程输出对比图)。
运行: python scripts/target_diagnose.py  (或双击 diagnose.bat;进度窗口+进度条)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config_loader import CFG
from load_data import load_complex, WORKBOOK_DIR
from forecast import targets_for, series_for_target
from run_test import eval_series, _make_progress as _mk_progress
from predict import holt_main_predict, formal_forecast, FORECAST_END
from predict import export_forecast, cleanup_charts, cleanup_output
from train_all import load_all_complexes
from drift import build_train_distribution, drift_assessment, drift_message

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_DIR = WORKBOOK_DIR          # data/test
OUTPUT_DIR = os.path.join(ROOT, "output")

# ---- 从集中配置读取诊断参数 ----
_DC = CFG["diagnose"]
N_DIAG_WINDOW = _DC["n_window"]    # 每个截断点预测月数
N_CUTOFFS = _DC["n_cutoffs"]       # 诊断截断点数


def suggest_cutoffs(old, n=N_CUTOFFS):
    """自动选取诊断截断点:从(起点+12月)到(终点-12月)均匀取 n 个。"""
    start = old["date"].min() + pd.offsets.DateOffset(months=12)
    end = old["date"].max() - pd.offsets.DateOffset(months=12)
    if end <= start:
        return [old["date"].max() - pd.offsets.DateOffset(months=12)]
    points = pd.date_range(start, end, periods=n)   # 均匀取 n 个(不指定 freq)
    return [p for p in points if p >= start]


def diagnose_one(c, events, pw=None, report=None):
    """对单个小区做多截断点诊断,返回 (诊断行, 汇总, {截断点: {序列: 预测df}})。"""
    name = c["name"]
    old = c["old"]
    cutoffs = suggest_cutoffs(old)
    rows = []
    cut_preds = {}
    for cut in cutoffs:
        train = old[old["date"] <= cut]
        # 预测窗口:(cut, cut+12月]
        win_end = cut + pd.offsets.DateOffset(months=N_DIAG_WINDOW)
        actual = old[(old["date"] > cut) & (old["date"] <= win_end)].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]
        preds_t = {}
        for t in targets_for(old):
            series = series_for_target(train, t)
            if len(series["price"].dropna()) < 8:
                continue
            df = holt_main_predict(series, N_DIAG_WINDOW)
            preds_t[t] = df
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            m = eval_series(actual[["date", col]].rename(columns={col: "price"}),
                            df, mask=actual_mask)
            if m:
                rows.append({
                    "截断点": cut.strftime("%Y-%m"), "序列": t,
                    "MAE": round(m["MAE"]), "MAPE%": round(m["MAPE%"], 2),
                    "得分": round(m["得分(0-100)"], 1),
                    "覆盖率%": round(m["区间覆盖率%"]),
                    "样本": m["样本数"],
                    "偏差%": round((m["预测P50均值"] / m["真实均值"] - 1) * 100, 1),
                })
        if preds_t:
            cut_preds[cut] = preds_t
    if not rows:
        return rows, None, cut_preds
    df = pd.DataFrame(rows)
    # 汇总(综合序列为主)
    comp = df[df["序列"] == "综合"]
    avg_score = comp["得分"].mean()
    avg_bias = comp["偏差%"].mean()
    spread = comp["得分"].max() - comp["得分"].min() if len(comp) > 1 else 0.0
    if avg_score >= 90 and spread <= 5:
        conf = "高"
    elif avg_score >= 75 or spread <= 15:
        conf = "中"
    else:
        conf = "低"
    summary = {
        "平均得分": round(avg_score, 1),
        "平均偏差%": round(avg_bias, 1),
        "窗口离散": round(spread, 1),
        "置信度": conf,
        "建议": ("Holt 预测稳定,正式预测可信" if conf == "高" else
                 "中等置信:正式预测请同时参考区间宽度" if conf == "中" else
                 "低置信:该小区波动大,正式预测仅作参考"),
    }
    return rows, summary, cut_preds


def main():
    from progress_gui import run_with_progress
    run_with_progress(main_impl, _mk_progress, title="目标小区诊断运行中")


def main_impl(pw=None):
    last_pct = [0]

    def report(text, pct, log=None):
        if pw is not None:
            pw.update(text, pct, log)
        else:
            if int(pct) >= last_pct[0] + 5 or pct >= 100:
                last_pct[0] = int(pct)
                print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    targets = sorted(f for f in os.listdir(TARGET_DIR)
                     if f.lower().endswith((".xlsx", ".xlsm")) and "宏观" not in f)
    if not targets:
        print("data/test 下没有目标小区,请先放入小区表格")
        return
    complexes = []
    for f in targets:
        try:
            complexes.append(load_complex(os.path.join(TARGET_DIR, f)))
        except Exception as e:
            print(f"  ⚠ 解析失败跳过: {f} ({e})")
    if not complexes:
        return
    cleanup_charts([c["name"] for c in complexes])
    cleanup_output()
    # 漂移检测:训练集行情分布
    drift_dist = None
    try:
        drift_dist = build_train_distribution(load_all_complexes(), None)
    except Exception as e:
        print(f"  ⚠ 漂移分布构建失败({e})")

    for ci, c in enumerate(complexes):
        name = c["name"]
        report(f"[{ci + 1}/{len(complexes)}] {name}:多截断点诊断", 4 + ci * 20,
               f"截断点自动选取(每窗口预测 {N_DIAG_WINDOW} 个月验证)")
        rows, summary, _ = diagnose_one(c, None, pw, report)
        if not rows:
            print(f"  ⚠ {name}: 历史不足,无法诊断")
            continue
        if drift_dist is not None:
            dr = drift_assessment(drift_dist, c["old"]["price"], None)
            print(drift_message(dr, name))
        print(f"\n=== {name} 诊断表(多截断点 walk-forward,严格无泄漏) ===")
        for r in rows:
            print(f"  截断 {r['截断点']} {r['序列']}: MAE={r['MAE']:>6,} "
                  f"MAPE={r['MAPE%']:5.2f}% 得分={r['得分']:5.1f} "
                  f"覆盖={r['覆盖率%']:3.0f}% 偏差={r['偏差%']:+5.1f}% (样本{r['样本']})")
        print(f"  → 汇总: 平均得分 {summary['平均得分']} | 平均偏差 {summary['平均偏差%']:+}% "
              f"| 窗口离散 {summary['窗口离散']} | 置信度:{summary['置信度']}")
        print(f"    建议: {summary['建议']}")

        # 正式预测(全部历史 → 2031.12,只输出预测表;图由 predict 流程统一处理)
        report(f"[{ci + 1}/{len(complexes)}] {name}:正式预测至 {FORECAST_END}", 20 + ci * 20,
               f"预测表 → output/predictions_{name}.xlsx")
        fresult = formal_forecast(c)
        export_forecast(name, fresult)
        fz = fresult["综合"]
        print(f"  正式预测(→{FORECAST_END}): 当前 {fz['P50'].iloc[0]:,.0f} → "
              f"{FORECAST_END} {fz['P50'].iloc[-1]:,.0f} ({fz['P50'].iloc[-1] / fz['P50'].iloc[0] - 1:+.1%})")

    report("全部完成", 100, "诊断表 + 正式预测产物已输出")
    if pw is not None:
        pw.done([f"诊断表 → output/diagnose_*.xlsx",
                 f"正式预测至 {FORECAST_END} → output/ + charts/"])


if __name__ == "__main__":
    main()
