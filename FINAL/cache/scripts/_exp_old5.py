# -*- coding: utf-8 -*-
"""对照:8/14 随机测试的 5 个小区,用当前正式模型(100轮×5模型+属性)再测一遍。
若分数与 8/14(88分)相当 → 模型没变差,11:47 的低分是抽样运气;
若也变差 → 训练集清理/其他系统因素。
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
from features import build_events, load_extra_events, load_macro, load_attrs
from predict import pooled_predict
from run_test import N_MONTHS, eval_series
from forecast import targets_for, series_for_target

OLD5 = ["天安曼哈顿_价格数据表", "康桥府_价格数据表", "玖里映月_价格数据表",
        "蠡湖香樟园(A区)_价格数据表", "中城誉品(南区)_价格数据表"]


def main():
    pkg = joblib.load(MODEL_FILE)
    models, meta = pkg["models"], pkg["meta"]
    attr_info = pkg.get("attr_info")
    macro = load_macro()
    attrs = load_attrs()
    complexes = load_all_complexes()
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None], ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    events = events[events.index <= TRAIN_CUTOFF]
    by_name = {c["name"]: c for c in complexes}
    meta_names = {m["name"]: m for m in meta}

    print("8/14 历史分数 → 当前正式模型(100轮×5模型+属性)分数:")
    for name in OLD5:
        if name not in by_name:
            print(f"  {name}: 不在训练集,跳过")
            continue
        c = by_name[name]
        train = c["old"][c["old"]["date"] <= TRAIN_CUTOFF]
        actual = c["old"][c["old"]["date"] > TRAIN_CUTOFF][["date", "price", "jingzhuang",
                                                            "maopi", "bieshu", "is_actual"]].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]
        if name in meta_names:
            cid, mean_price = meta_names[name]["id"], meta_names[name]["mean"]
        else:
            cid, mean_price = 10000, float(train["price"].mean())
        maes, scores, biases = [], [], []
        for t in targets_for(c["old"]):
            series = series_for_target(train, t)
            rd = pooled_predict(series, mean_price, cid, models, events, N_MONTHS,
                                macro=macro, attrs=attrs, attr_info=attr_info, complex_name=name)
            if rd is None:
                continue
            q = rd.drop(columns=["date"])
            df = pd.DataFrame({"date": rd["date"],
                               "P10": q.quantile(0.10, axis=1), "P50": q.quantile(0.50, axis=1),
                               "P90": q.quantile(0.90, axis=1), "mean": q.mean(axis=1)})
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            m = eval_series(actual[["date", col]].rename(columns={col: "price"}), df, mask=actual_mask)
            if m:
                maes.append(m["MAE"]); scores.append(m["得分(0-100)"])
                biases.append(m["预测P50均值"] / m["真实均值"] - 1)
        if maes:
            print(f"  {name[:12]}: MAE={np.mean(maes):,.0f} 得分={np.mean(scores):.1f} "
                  f"偏差={np.mean(biases):+.1%}")


if __name__ == "__main__":
    main()
