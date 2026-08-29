# -*- coding: utf-8 -*-
"""
predict_random.py — 从 ALL 随机抽取 N 个小区,直接预测(不训练):
  1. 加载已训练的联合模型(models/ensemble_all.joblib);
  2. 从 数据快照/ALL 筛选"有 2024.09 后真实数据"的小区,随机抽 N 个;
  3. 每小区:截断 2024.08 → 联合模型递归预测 22 个月(2024.09–2026.06);
  4. 输出:历史折线图(charts/,开盘至今)、预测对比折线图(对比图/{编号}/)、
     逐月对比表,并追加打分表(accuracy_metrics.xlsx / 打分表.xlsx,[随机]前缀)。

运行: python scripts/predict_random.py  (弹窗 + 进度条)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from train_all import load_all_complexes, MODEL_FILE, TRAIN_CUTOFF
from features import build_events, load_extra_events, load_macro, load_attrs
from forecast import targets_for, series_for_target
from run_test import (N_MONTHS, eval_series, calc_bias_ratio, calibration_factor,
                      USE_CALIBRATION, append_metrics, append_score_table,
                      export_comparison, plot_comparison, OUTPUT_DIR, METRICS_FILE,
                      next_run_folder, _make_progress as _mk_progress)
from predict import pooled_predict
from features import ATTRIBUTE_COLS
from plot_monthly import draw as draw_history

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHART_DIR = os.path.join(ROOT, "comparison")
OFFICIAL_CHART_DIR = os.path.join(ROOT, "charts")
os.makedirs(OFFICIAL_CHART_DIR, exist_ok=True)

N_SAMPLE = 5        # 随机抽取小区数
SEED = None         # None=真随机(每次不同);填数字可复现


def pick_random(complexes, n=N_SAMPLE, seed=SEED):
    """筛选有 2024.09 后真实数据的小区,随机抽 n 个。"""
    valid = []
    for c in complexes:
        old = c["old"]
        if len(old) == 0:
            continue
        after = old[old["date"] > TRAIN_CUTOFF]
        train_part = old[old["date"] <= TRAIN_CUTOFF]
        if len(after) == 0 or after["price"].isna().all():
            continue
        if len(train_part) < 12:     # 训练段至少 12 个月:滞后12月价格可计算(缺失收益由兜底置0)
            continue
        valid.append(c)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    return [valid[i] for i in idx], len(valid)


def main():
    """独立运行入口:进度窗口在子线程,主线程刷新(窗口不"未响应")。"""
    from progress_gui import run_with_progress
    run_with_progress(main_impl, _mk_progress, title="随机小区预测运行中")


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

    if not os.path.exists(MODEL_FILE):
        print("未找到联合模型,请先运行 训练.bat")
        return
    pkg = joblib.load(MODEL_FILE)
    models, meta = pkg["models"], pkg["meta"]
    report(f"加载联合模型:{len(models)} 个集成模型", 2, f"cfg={pkg.get('cfg', '标准')}")

    complexes = load_all_complexes()
    # 事件:ALL 小区事件 + 外部事件,截断到 2024.08
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                       ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    events = events[events.index <= TRAIN_CUTOFF]
    report(f"事件截断:{len(events)} 条(≤2024.08)", 3)
    macro = load_macro()
    attrs = load_attrs()
    attr_info = pkg.get('attr_info')
    need_attrs = any(c in pkg.get('feature_cols', []) for c in ATTRIBUTE_COLS)
    if need_attrs and attrs is None:
        print('⚠ 模型训练时含属性特征,但属性表不可用,本次预测取消')
        return
    # 一致性检查:模型训练时若含宏观特征,推理必须提供宏观数据(列数不匹配会崩)
    need_macro = "lpr" in pkg.get("feature_cols", [])
    if need_macro and not macro:
        print("⚠ 模型训练时包含宏观特征,但宏观数据不可用,本次预测取消")
        print("  请确认 数据快照/无锡宏观月度数据.xlsx 存在且可读")
        return
    report(f"宏观特征:{'已加载(LPR+全市均价)' if macro else '不参与(模型未含宏观)'}", 3)

    picked, total_valid = pick_random(complexes)
    report(f"随机抽取 {len(picked)} 个小区(可打分候选 {total_valid} 个)", 4,
           f"抽取结果: {[c['name'] for c in picked]}")

    meta_names = {m["name"]: m for m in meta}
    run_folder = next_run_folder()
    run_time = time.strftime("%Y-%m-%d %H:%M")
    metrics_rows = []
    summaries = []
    total_steps = len(picked) * 3
    done = [0]

    for ci, c in enumerate(picked):
        name = c["name"]
        old = c["old"]
        train = old[old["date"] <= TRAIN_CUTOFF]
        actual = old[old["date"] > TRAIN_CUTOFF][["date", "price", "jingzhuang",
                                                  "maopi", "bieshu", "is_actual"]].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]

        # 1) 历史折线图(开盘至今)
        report(f"[{ci + 1}/{len(picked)}] {name}:历史折线图", 5 + ci * 2, f"{name}: 开盘至今走势")
        draw_history(name, c)

        # 2) 联合模型递归预测(不训练)
        if name in meta_names:
            cid, mean_price = meta_names[name]["id"], meta_names[name]["mean"]
        else:
            max_id = max(m["id"] for m in meta) if meta else -1
            cid = max_id + 1 + ci          # 修复:新小区 id 不与训练小区冲突
            mean_price = float(train["price"].mean())
        report(f"[{ci + 1}/{len(picked)}] {name}:递归预测", 6 + ci * 2,
               f"complex_id={cid}, 标准化均值 {mean_price:,.0f}")

        preds = {}
        for t in targets_for(old):
            series = series_for_target(train, t)
            rounds_df = pooled_predict(series, mean_price, cid, models, events, N_MONTHS,
                                       macro=macro, attrs=attrs, attr_info=attr_info,
                                       complex_name=name)
            if rounds_df is None:
                print(f"  ⚠ {t}: 2024.08 前历史不足 12 个月,跳过该序列")
                continue
            preds[t] = pd.DataFrame({
                "date": rounds_df["date"],
                "P10": rounds_df.drop(columns=["date"]).quantile(0.10, axis=1),
                "P50": rounds_df.drop(columns=["date"]).quantile(0.50, axis=1),
                "P90": rounds_df.drop(columns=["date"]).quantile(0.90, axis=1),
                "mean": rounds_df.drop(columns=["date"]).mean(axis=1),
            })
            done[0] += 1
            report(f"[{ci + 1}/{len(picked)}] {name}:预测 {t}",
                   8 + 78 * done[0] / total_steps, None)
        if not preds:
            print(f"  ⚠ {name}: 所有序列历史均不足,跳过该小区")
            continue

        # 3) 校准(自动开关)
        bias = calc_bias_ratio(train, events)
        if USE_CALIBRATION and abs(bias) <= 0.08:
            calib = calibration_factor(bias)
            print(f"\n[{name}] 偏差校准:训练期偏差 {bias:+.2%} → 校准系数 {calib:.4f}")
            for t in preds:
                for col in ("P10", "P50", "P90", "mean"):
                    preds[t][col] = preds[t][col] * calib
        else:
            print(f"\n[{name}] 本次未校准: 训练期偏差 {bias:+.2%}(>8% 自动跳过或已关闭)")

        # 4) 打分 + 输出
        print(f"\n[{name}] 联合模型预测 2024.09–2026.06 与真实对比:")
        scores = []
        for t, df in preds.items():
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            act = actual[["date", col]].rename(columns={col: "price"})
            m = eval_series(act, df, mask=actual_mask)
            if m:
                print(f"  {t}: MAE={m['MAE']:,.0f}  MAPE={m['MAPE%']:.2f}%  "
                      f"得分={m['得分(0-100)']:.1f}  覆盖率={m['区间覆盖率%']:.0f}%  "
                      f"(样本{m['样本数']}个真实点)")
                metrics_rows.append([run_time, f"[随机]{name}", t,
                                     round(m["MAE"]), round(m["RMSE"]), round(m["MAPE%"], 2),
                                     round(m["得分(0-100)"], 1), round(m["区间覆盖率%"]),
                                     m["样本数"], round(m["真实均值"]), round(m["预测P50均值"])])
                scores.append(m["得分(0-100)"])
        if scores:
            metrics_rows.append([run_time, f"[随机]{name}", "平均",
                                 "", "", "", round(sum(scores) / len(scores), 1),
                                 "", len(scores), "", ""])

        # 5) 对比表 + 对比折线图
        export_comparison(name, actual, preds)
        plot_comparison(name, actual, preds,
                        os.path.join(run_folder, f"accuracy_{name}.png"), actual_mask)
        report(f"[{ci + 1}/{len(picked)}] {name}:对比图完成", 88 + ci * 2,
               f"对比图 → {run_folder}")

        last_real = actual["price"].iloc[-1]
        last_pred = preds["综合"].iloc[-1]["P50"]
        summaries.append((name, last_real, last_pred))

    if metrics_rows:
        append_metrics(metrics_rows, {})
    append_score_table()
    report("全部完成", 100, f"打分表已追加 → {METRICS_FILE}")

    print("\n" + "=" * 90)
    print("随机 5 小区最终对比(2026.06 末月): 真实 vs 预测P50")
    for name, real, pred in summaries:
        print(f"  {name}: 真实 {real:,.0f}  预测 {pred:,.0f}  偏差 {pred - real:+,.0f} "
              f"({pred / real - 1:+.1%})")
    print(f"\n全部完成:历史折线图 → {OFFICIAL_CHART_DIR}  对比图 → {run_folder}")
    if pw is not None:
        pw.done([f"随机 {len(picked)} 小区:打分表已追加",
                 f"历史折线图 → {OFFICIAL_CHART_DIR}",
                 f"预测对比图 → {run_folder}"])


if __name__ == "__main__":
    main()
