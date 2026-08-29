# -*- coding: utf-8 -*-
"""定位实验:随机测试变差的元凶(5模型?属性?训练集清理?)
配置 A: 3模型(旧) + 无属性 25特征   ← 与 8/14 基线相同(除训练集清理)
配置 B: 5模型(新) + 无属性 25特征
配置 C: 5模型(新) + 属性 39特征     ← 当前正式配置
固定 5 个小区(与 11:47 随机测试相同)打分对比。
"""
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from train_all import load_all_complexes, build_pooled, TRAIN_CUTOFF
from features import build_events, load_extra_events, load_macro, load_attrs
from predict import pooled_predict
from run_test import N_MONTHS, eval_series
from forecast import targets_for, series_for_target

R = 20
NT = 150
NAMES = ["红豆首府_价格数据表", "红豆清华苑_价格数据表", "中梁中奥溪涧堂_价格数据表",
         "蠡湖香樟园(A区)_价格数据表", "中南君悦府_价格数据表"]


class Ens:
    def __init__(self, models):
        self.models = models
        self.weights = np.ones(len(models)) / len(models)

    def fit(self, X, y, w=None):
        for _, m in self.models:
            m.fit(X, y, sample_weight=w)
        return self

    def predict(self, X):
        return np.tensordot(self.weights, [m.predict(X) for _, m in self.models], axes=(0, 0))


def make_models(cfg):
    common = dict(n_estimators=NT, max_depth=4, min_samples_leaf=3, random_state=42)
    gbr = GradientBoostingRegressor(n_estimators=NT, max_depth=4, learning_rate=0.03,
                                    subsample=0.9, validation_fraction=0.1,
                                    n_iter_no_change=20, random_state=42)
    if cfg == "A":
        return [("Ridge", Ridge(alpha=5.0)),
                ("RF", RandomForestRegressor(**common)), ("GBR", gbr)]
    xgb = XGBRegressor(n_estimators=NT, max_depth=4, learning_rate=0.03, subsample=0.9,
                       colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.5, random_state=42)
    lgbm = LGBMRegressor(n_estimators=NT, max_depth=4, learning_rate=0.03, subsample=0.9,
                         colsample_bytree=0.9, reg_lambda=1.0, reg_alpha=0.5,
                         random_state=42, verbose=-1)
    return [("Ridge", Ridge(alpha=5.0)), ("RF", RandomForestRegressor(**common)),
            ("GBR", gbr), ("XGB", xgb), ("LGBM", lgbm)]


def train(Xv, yv, wv, cfg):
    rng = np.random.default_rng(2026)
    out = []
    for _ in range(R):
        idx = rng.choice(len(Xv), size=len(Xv), replace=True)
        e = Ens(make_models(cfg))
        e.fit(Xv[idx], yv[idx], wv[idx])
        out.append(e)
    return out


def test(c, models, events, macro, attrs, attr_info, Xcols_has_attr):
    name = c["name"]
    old = c["old"]
    train = old[old["date"] <= TRAIN_CUTOFF]
    actual = old[old["date"] > TRAIN_CUTOFF][["date", "price", "jingzhuang",
                                              "maopi", "bieshu", "is_actual"]].copy()
    actual["date"] = actual["date"].dt.strftime("%Y-%m")
    actual_mask = actual.set_index("date")["is_actual"]
    cid = 10000
    scores = []
    for t in targets_for(old):
        series = series_for_target(train, t)
        rd = pooled_predict(series, float(train["price"].mean()), cid, models, events,
                            N_MONTHS, macro=macro,
                            attrs=(attrs if Xcols_has_attr else None),
                            attr_info=(attr_info if Xcols_has_attr else None),
                            complex_name=name)
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
            scores.append(m["MAE"])
    return np.mean(scores) if scores else float("nan")


def main():
    t0 = time.time()
    complexes = load_all_complexes()
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None], ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    events = events[events.index <= TRAIN_CUTOFF]
    macro = load_macro()
    attrs = load_attrs()
    X_a, y, w, meta, _, _ = build_pooled(complexes, events, macro, attrs=None)
    X_c, _, _, _, _, attr_info = build_pooled(complexes, events, macro, attrs=attrs)
    print(f"面板: 无属性 {X_a.shape[1]} 特征 / 有属性 {X_c.shape[1]} 特征,{len(meta)} 小区")

    by_name = {c["name"]: c for c in complexes}
    for cfg, X in (("A", X_a), ("B", X_a), ("C", X_c)):
        models = train(X.values, y.values, w.values, cfg)
        maes = []
        for name in NAMES:
            if name not in by_name:
                print(f"  {cfg} {name}: 不在训练集,跳过")
                continue
            m = test(by_name[name], models, events, macro, attrs, attr_info, cfg == "C")
            maes.append(m)
            print(f"  {cfg} {name[:12]}: MAE={m:,.0f}")
        if maes:
            print(f"  {cfg} 平均 MAE: {np.mean(maes):,.0f}  耗时 {time.time()-t0:.0f}s\n")


if __name__ == "__main__":
    main()
