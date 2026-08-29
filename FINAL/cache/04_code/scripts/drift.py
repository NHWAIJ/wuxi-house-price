# -*- coding: utf-8 -*-
"""
drift.py — 分布漂移检测器:
预测时判断"目标小区当前行情是否在训练集分布内"。若落在训练分布尾部
(比绝大多数训练小区跌得更多/反弹更猛),说明模型从未见过这种行情,
外推风险高 → 自动降级提示"此预测置信度低"。

指标(全部只用 ≤ 预测时刻的信息,无泄漏):
  - 近 12 个月动量(当前价/12个月前 - 1)在训练集的分位数
  - 回撤比(当前价/历史峰值)在训练集的分位数
  - 宏观动量(全市均价 12 个月变化)的分位数
漂移等级 = 距最近分布边缘的距离: <5% 高, <15% 中, 否则低。
"""
import numpy as np
import pandas as pd


def build_train_distribution(complexes, macro=None):
    """
    用训练集构建"行情分布":每个训练小区最后时点的
    (12月动量, 回撤比) + 宏观全市均价 12 月动量(如提供)。
    返回 dict 含各指标的数组。
    """
    mom12s, dds = [], []
    for c in complexes:
        s = c["old"]["price"].dropna()
        if len(s) < 13:
            continue
        mom12s.append(s.iloc[-1] / s.iloc[-13] - 1)
        peak = float(s.max())
        dds.append(s.iloc[-1] / peak if peak else 1.0)
    out = {"mom12": np.array(mom12s), "dd": np.array(dds)}
    if macro and "全市二手均价_聚合" in macro:
        cp = macro["全市二手均价_聚合"]["全市均价(元/㎡)"]
        if len(cp) >= 13:
            out["city_mom12"] = np.array([cp.iloc[-1] / cp.iloc[-13] - 1])
    return out


def _pct(x, dist):
    """x 在 dist 中的百分位(0-100)。"""
    return float((dist < x).mean() * 100)


def drift_assessment(dist, target_series, macro=None):
    """
    评估目标小区当前漂移等级。
    dist: build_train_distribution() 结果;target_series: 该小区 price Series。
    返回 dict: {level, score, details:[...]}。score 0-100,越低越异常。
    """
    s = target_series.dropna()
    if len(s) < 13 or len(dist.get("mom12", [])) < 10:
        return {"level": "未知", "score": 50.0, "details": ["样本不足,无法评估漂移"]}
    mom12 = float(s.iloc[-1] / s.iloc[-13] - 1)
    dd = float(s.iloc[-1] / s.max())
    p_mom = _pct(mom12, dist["mom12"])
    p_dd = _pct(dd, dist["dd"])
    details = [
        f"近12月动量 {mom12:+.1%} → 训练集 {p_mom:.0f}% 分位",
        f"回撤比 {dd:.2f}(当前/峰值) → 训练集 {p_dd:.0f}% 分位",
    ]
    p_city = None
    if macro and "city_mom12" in dist and "全市二手均价_聚合" in macro:
        cp = macro["全市二手均价_聚合"]["全市均价(元/㎡)"]
        if len(cp) >= 13:
            city_mom = float(cp.iloc[-1] / cp.iloc[-13] - 1)
            p_city = _pct(city_mom, dist["city_mom12"])
            details.append(f"全市均价12月动量 {city_mom:+.1%} → 训练集 {p_city:.0f}% 分位")
    edges = [min(p_mom, 100 - p_mom), min(p_dd, 100 - p_dd)]
    if p_city is not None:
        edges.append(min(p_city, 100 - p_city))
    edge = min(edges)
    score = float(edge)
    if edge < 5:
        level = "高"
    elif edge < 15:
        level = "中"
    else:
        level = "低"
    return {"level": level, "score": score, "details": details}


def drift_message(res, name="该小区"):
    """漂移结果的展示文案。"""
    lv = res["level"]
    if lv == "未知":
        return f"[漂移] {name}: 样本不足,无法评估"
    if lv == "低":
        return (f"[漂移] {name}: 当前行情在训练分布内(边缘度 {res['score']:.0f}%),"
                f"预测可信度正常")
    if lv == "中":
        return (f"[漂移] {name}: 当前行情接近训练分布边缘({res['score']:.0f}%),"
                f"预测需谨慎解读。{'；'.join(res['details'])}")
    return (f"[漂移] {name}: 当前行情显著偏离训练分布(边缘度 {res['score']:.0f}% —— "
            f"比 {100 - res['score']:.0f}% 的训练小区都更极端),模型外推风险高,"
            f"此预测仅作参考。{'；'.join(res['details'])}")
