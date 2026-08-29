# -*- coding: utf-8 -*-
"""实验:ALL 小区用 Holt vs 联合模型,哪个更好?
抽 10 个有 2024.09+ 真实数据的小区,两种方法在同一测试任务(截断2024.08→预测22月)打分。
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from train_all import load_all_complexes, MODEL_FILE, TRAIN_CUTOFF
from features import build_events, load_extra_events, load_macro
from predict import pooled_predict, holt_main_predict
from forecast import targets_for, series_for_target
from run_test import N_MONTHS, eval_series

SEED = 42
N_SAMPLE = 10


def main():
    pkg = joblib.load(MODEL_FILE)
    models, meta = pkg["models"], pkg["meta"]
    meta_names = {m["name"]: m for m in meta}
    macro = load_macro()

    complexes = load_all_complexes()
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                       ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    events = events[events.index <= TRAIN_CUTOFF]

    # 筛选有真实测试数据的小区
    valid = []
    for c in complexes:
        old = c["old"]
        after = old[old["date"] > TRAIN_CUTOFF]
        train_part = old[old["date"] <= TRAIN_CUTOFF]
        if len(after) == 0 or after["price"].isna().all():
            continue
        if len(train_part) < 12:
            continue
        valid.append(c)
    rng = np.random.default_rng(SEED)
    picked = [valid[i] for i in rng.choice(len(valid), size=min(N_SAMPLE, len(valid)), replace=False)]
    print(f"抽取 {len(picked)} 个小区(种子 {SEED})")

    rows = []
    for ci, c in enumerate(picked):
        name = c["name"]
        old = c["old"]
        train = old[old["date"] <= TRAIN_CUTOFF]
        actual = old[old["date"] > TRAIN_CUTOFF][["date", "price", "jingzhuang",
                                                  "maopi", "bieshu", "is_actual"]].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]

        if name in meta_names:
            cid, mean_price = meta_names[name]["id"], meta_names[name]["mean"]
        else:
            cid = max(m["id"] for m in meta) + 1 + ci
            mean_price = float(train["price"].mean())

        h_maes, j_maes, h_sc, j_sc = [], [], [], []
        for t in targets_for(old):
            series = series_for_target(train, t)
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            act = actual[["date", col]].rename(columns={col: "price"})
            # Holt
            try:
                hdf = holt_main_predict(series, N_MONTHS)
                mh = eval_series(act, hdf, mask=actual_mask)
            except Exception:
                mh = None
            # 联合模型
            rd = pooled_predict(series, mean_price, cid, models, events, N_MONTHS, macro=macro)
            if rd is None:
                continue
            q = rd.drop(columns=["date"])
            jdf = pd.DataFrame({"date": rd["date"],
                                "P10": q.quantile(0.10, axis=1),
                                "P50": q.quantile(0.50, axis=1),
                                "P90": q.quantile(0.90, axis=1),
                                "mean": q.mean(axis=1)})
            mj = eval_series(act, jdf, mask=actual_mask)
            if mh and mj:
                h_maes.append(mh["MAE"]); j_maes.append(mj["MAE"])
                h_sc.append(mh["得分(0-100)"]); j_sc.append(mj["得分(0-100)"])
        if h_maes:
            hm, jm = np.mean(h_maes), np.mean(j_maes)
            hs, js = np.mean(h_sc), np.mean(j_sc)
            winner = "Holt" if hm < jm else "联合模型"
            print(f"  {name[:18]:20s} Holt MAE={hm:7,.0f}({hs:4.1f}分)  |  联合 MAE={jm:7,.0f}({js:4.1f}分)  |  {winner}胜")
            rows.append((name, hm, jm, hs, js))

    print("\n" + "=" * 80)
    if rows:
        h_all = np.mean([r[1] for r in rows]); j_all = np.mean([r[2] for r in rows])
        h_s = np.mean([r[3] for r in rows]); j_s = np.mean([r[4] for r in rows])
        print(f"平均: Holt MAE={h_all:,.0f}({h_s:.1f}分)  vs  联合模型 MAE={j_all:,.0f}({j_s:.1f}分)")
        h_wins = sum(1 for r in rows if r[1] < r[2])
        print(f"Holt 胜 {h_wins}/{len(rows)} 个小区")


if __name__ == "__main__":
    main()
