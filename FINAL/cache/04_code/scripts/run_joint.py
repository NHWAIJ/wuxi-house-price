# -*- coding: utf-8 -*-
"""
run_joint.py — 单进度窗口全流程:联合训练(0-50%)→ 自动预测(50-100%)。

双击 训练.bat 即调用本脚本:一个窗口走完全程,进度条持续更新(不再双窗口卡顿)。
训练完成自动预测(对原两个小区 2024.09 起测试)→ 打分表追加 + 对比折线图。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from train_all import (load_all_complexes, build_pooled, MODEL_FILE, MODEL_DIR,
                       _make_progress, TRAIN_ROUNDS, HIGH_CFG, HALF_LIFE,
                       train_parallel, N_JOBS)
from features import build_events, load_extra_events, load_macro, load_attrs
from models import Ensemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_phase(complexes, events, pw, macro=None):
    """联合训练 30 轮,返回模型列表。进度 0-50%。"""
    last_pct = [0]

    def report(text, pct, log=None):
        pct = pct * 0.5          # 训练阶段占 0-50%
        if pw is not None:
            pw.update(text, pct, log)
        else:
            if int(pct) >= last_pct[0] + 5 or pct >= 100:
                last_pct[0] = int(pct)
                print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    report(f"解析 {len(complexes)} 个小区", 4)
    attrs = load_attrs()
    X, y, w, meta, x_cols, attr_info = build_pooled(complexes, events, macro, attrs)
    report(f"面板样本 {len(X)} 行 × {X.shape[1]} 特征,{len(meta)} 个小区"
           f"(近期加权,半衰期 {HALF_LIFE} 个月)", 10,
           f"训练小区: {[m['name'] for m in meta][:5]} …")
    Xv, yv, wv = X.values, y.values, w.values
    report(f"轮间并行训练:{N_JOBS} 进程 × {TRAIN_ROUNDS} 轮(CPU 拉满)", 12)
    models = train_parallel(Xv, yv, wv, TRAIN_ROUNDS, HIGH_CFG,
                            on_done=lambda d, t: report(
                                f"训练中:{d}/{t} 轮(并行)", 12 + 82 * d / t, None))
    joblib.dump({"models": models, "meta": meta,
                 "feature_cols": x_cols, "has_macro": bool(macro),
                 "attr_info": attr_info,
                 "n_rounds": TRAIN_ROUNDS, "cfg": HIGH_CFG,
                 "trained_at": time.strftime("%Y-%m-%d %H:%M")},
                MODEL_FILE)
    report("训练完成,模型已保存", 98, f"模型 → {MODEL_FILE}")
    return models


def main():
    from progress_gui import run_with_progress
    run_with_progress(_run, _make_progress, title="联合训练运行中")


def _run(pw):
    def report(text, pct, log=None):
        if pw is not None:
            pw.update(text, pct, log)
        else:
            print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    t0 = time.time()
    complexes = load_all_complexes()
    if not complexes:
        print(f"data/train 下未找到可解析的小区数据")
        return

    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                       ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    report(f"事件构建:{len(events)} 条", 2)
    macro = load_macro()     # 训练与预测共用同一份宏观数据
    report(f"宏观特征:{'已加载(LPR+全市均价)' if macro else '不可用'}", 2)

    # 训练(进度 0-50%)
    train_phase(complexes, events, pw, macro)

    # 预测(进度 50-100%,复用同一窗口)
    print("\n→ 开始预测(对原两个小区 2024.09 起测试预测)…\n")
    from predict import main_impl as predict_main
    predict_main(pw=pw, events=events, macro=macro, base_pct=50.0, span=50.0)

    print(f"\n全程完成,总耗时 {time.time() - t0:.0f} 秒")
    if pw is not None:
        pw.done([f"联合模型 → {MODEL_FILE}",
                 "打分表已追加(accuracy_metrics.xlsx / 打分表.xlsx)",
                 "对比折线图 → 对比图/{编号}/"])


if __name__ == "__main__":
    main()
