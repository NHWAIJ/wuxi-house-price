# -*- coding: utf-8 -*-
"""A/B 实验:Holt vs ARIMA —— 同一数据、同一测试条件(截断2024.08→预测22月→打分)。
导师建议 ARIMA;用 walk-forward 式测试打分回答"哪个模型在当前数据上更准"。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from load_data import load_complex, WORKBOOK_DIR
from forecast import targets_for, series_for_target
from run_test import CUTOFF, N_MONTHS, eval_series
from predict import holt_main_predict

SCORES = {}


def arima_predict(series, n, order=None):
    """ARIMA 预测(自动选阶 d=1, p,q 按 AIC 从 {0,1,2}² 选)+ 同样的季节叠加。"""
    from statsmodels.tsa.arima.model import ARIMA
    y = series["price"].values
    best, best_aic = None, np.inf
    if order is None:
        for p in (0, 1, 2):
            for q in (0, 1, 2):
                try:
                    m = ARIMA(y, order=(p, 1, q)).fit(method_kwargs={"maxiter": 200})
                except Exception:
                    continue
                if m.aic < best_aic:
                    best, best_aic = (p, q), m.aic
        order = (best[0], 1, best[1])
    m = ARIMA(y, order=order).fit()
    pred = np.asarray(m.forecast(n))
    resid = y - np.asarray(m.fittedvalues)
    rstd = float(np.std(resid, ddof=1))
    # 季节叠加(与 Holt 版一致)
    s_tmp = series.copy()
    s_tmp["_m"] = s_tmp["date"].dt.month
    s_tmp["_r"] = resid
    season = s_tmp.groupby("_m")["_r"].mean().rolling(3, min_periods=1, center=True).mean()
    future_dates = pd.date_range(series["date"].max() + pd.offsets.MonthEnd(1),
                                 periods=n, freq="ME")
    seas_adj = np.array([season.get(m, 0.0) for m in future_dates.month])
    p50 = pred + seas_adj
    return pd.DataFrame({
        "date": future_dates.strftime("%Y-%m"),
        "P10": p50 - 1.28 * rstd, "P50": p50, "P90": p50 + 1.28 * rstd, "mean": p50,
    }), order


def run():
    files = sorted(f for f in os.listdir(WORKBOOK_DIR)
                   if f.lower().endswith((".xlsx", ".xlsm")) and "宏观" not in f)
    for f in files:
        c = load_complex(os.path.join(WORKBOOK_DIR, f))
        name = c["name"]
        old = c["old"]
        train = old[old["date"] <= CUTOFF]
        actual = old[old["date"] > CUTOFF][["date", "price", "jingzhuang",
                                            "maopi", "bieshu", "is_actual"]].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]
        print(f"\n=== {name} ===")
        for t in targets_for(old):
            series = series_for_target(train, t)
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            act = actual[["date", col]].rename(columns={col: "price"})
            # Holt(当前方案)
            h = holt_main_predict(series, N_MONTHS)
            mh = eval_series(act, h, mask=actual_mask)
            # ARIMA
            try:
                a, order = arima_predict(series, N_MONTHS)
                ma = eval_series(act, a, mask=actual_mask)
            except Exception as e:
                print(f"  {t}: ARIMA 失败 {str(e)[:60]}")
                continue
            if mh and ma:
                print(f"  {t}: Holt MAE={mh['MAE']:6,.0f} 得分={mh['得分(0-100)']:5.1f}"
                      f"  |  ARIMA(1,1,{order[1]}) MAE={ma['MAE']:6,.0f} 得分={ma['得分(0-100)']:5.1f}"
                      f"  |  {'Holt胜' if mh['MAE'] < ma['MAE'] else 'ARIMA胜'}")
                SCORES.setdefault(name, []).append(("holt", mh["MAE"]))
                SCORES[name].append(("arima", ma["MAE"]))

    print("\n" + "=" * 70)
    for name, lst in SCORES.items():
        h = np.mean([v for k, v in lst if k == "holt"])
        a = np.mean([v for k, v in lst if k == "arima"])
        print(f"{name}: Holt 平均MAE {h:,.0f}  vs  ARIMA 平均MAE {a:,.0f}"
              f"  → {'Holt 更准' if h < a else 'ARIMA 更准'} (差 {abs(h - a):,.0f})")


if __name__ == "__main__":
    run()
