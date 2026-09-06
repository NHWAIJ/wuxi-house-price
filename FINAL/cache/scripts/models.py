# -*- coding: utf-8 -*-
"""
models.py — 集成模型 + 统计基线 + 验证 + 多轮训练预测。

模型:
  集成 = Ridge + RandomForest + GradientBoosting 的均值(小样本下的推荐组合);
  基线 = Holt 指数平滑(statsmodels),用于对照集成模型的增益。
验证:expanding walk-forward(每 6 个月一切割点,预测后 12 个月);
      稀疏小区(帝泊湾)只在真实数据点上计误差,保证指标诚实。
多轮训练/多轮预测:bootstrap 有放回抽样训练集,每轮三模型集成递归预测,
      汇总 P10/P50/P90 给出区间(而不是单次点估计)。
"""
import numpy as np
import pandas as pd

from config_loader import CFG
from features import FEATURE_COLS, build_full_features, future_event_flow

# ---- 从集中配置读取模型参数 ----
_MC = CFG["models"]
HORIZON = _MC["horizon"]
N_ROUNDS = _MC["n_rounds"]
WALK_STEP = _MC["walk_step"]


# ---------------------------------------------------------------- 模型

def get_models(n_estimators=400, rf_depth=4, gbr_depth=3, lr=0.03, min_leaf=4,
               n_jobs=-1):
    """默认=标准配置;联合训练可传入高强度配置(rf_depth/gbr_depth/lr/min_leaf)。
    GBR 带早停(validation_fraction + n_iter_no_change),防过拟合并自动确定有效树数。
    XGB/LGBM(导师推荐):实验(2026-08)证明加入后联合模型得分 73.4→81.0,已并入;
    未安装 lightgbm/xgboost 的环境自动跳过(其余模型照常工作)。
    n_jobs: RF 并行线程数(-1=全部核)。轮间并行训练时传 1(进程数已占满核,
    避免线程超订;GBR 本身串行无此参数)。"""
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    models = [
        ("Ridge", Ridge(alpha=5.0)),
        ("RF", RandomForestRegressor(n_estimators=n_estimators, max_depth=rf_depth,
                                     min_samples_leaf=min_leaf, random_state=42,
                                     n_jobs=n_jobs)),
        ("GBR", GradientBoostingRegressor(n_estimators=n_estimators, max_depth=gbr_depth,
                                          learning_rate=lr, subsample=0.9,
                                          validation_fraction=0.1, n_iter_no_change=20,
                                          tol=1e-4, random_state=42)),
    ]
    # XGB/LGBM:树数减半控制模型体积,其余与 GBR 同深度/学习率
    for mname, mk in (("XGB", None), ("LGBM", None)):
        try:
            if mname == "XGB":
                from xgboost import XGBRegressor
                mk = XGBRegressor(n_estimators=n_estimators // 2, max_depth=gbr_depth,
                                  learning_rate=lr, subsample=0.9, colsample_bytree=0.9,
                                  reg_lambda=1.0, reg_alpha=0.5, random_state=42)
            else:
                from lightgbm import LGBMRegressor
                mk = LGBMRegressor(n_estimators=n_estimators // 2, max_depth=gbr_depth,
                                   learning_rate=lr, subsample=0.9, colsample_bytree=0.9,
                                   reg_lambda=1.0, reg_alpha=0.5, random_state=42,
                                   verbose=-1)
            models.append((mname, mk))
        except ImportError:
            pass
    return models


class Ensemble:
    """
    Ridge + RF + GBR 集成:预测 = 三模型加权平均。
    weights: 默认 (1/3, 1/3, 1/3) 等权;可调大线性模型(Ridge)权重以增强趋势外推
    (对单边行情更贴切),但权重过高会降低树模型的历史形态捕捉能力——以 walk-forward
    验证指标为准,避免过拟合某一类行情。
    """

    def __init__(self, weights=None, n_estimators=400, rf_depth=4, gbr_depth=3,
                 lr=0.03, min_leaf=4, n_jobs=-1):
        self.models = get_models(n_estimators=n_estimators, rf_depth=rf_depth,
                                 gbr_depth=gbr_depth, lr=lr, min_leaf=min_leaf,
                                 n_jobs=n_jobs)
        if weights is None:
            # 自动均分给实际模型数(修复:默认 1/3 硬编码,加 XGB/LGBM 后变 5 个模型会失配)
            weights = np.ones(len(self.models)) / len(self.models)
        self.weights = np.asarray(weights, dtype=float)
        self.weights = self.weights / self.weights.sum()

    def fit(self, X, y, sample_weight=None):
        for _, m in self.models:
            m.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        preds = np.array([m.predict(X) for _, m in self.models])
        return np.tensordot(self.weights, preds, axes=(0, 0))


def fit_holt(series):
    """Holt 线性趋势指数平滑(自动优化 α/β),返回已拟合对象。"""
    from statsmodels.tsa.holtwinters import Holt
    if len(series) < 8:
        raise ValueError("Holt 需要至少 8 个观测")
    return Holt(series).fit(optimized=True)


def holt_forecast(series, n):
    """Holt 外推 n 个月(统计基线;不吸收事件)。"""
    fit = fit_holt(series)
    return np.asarray(fit.forecast(n))


# ---------------------------------------------------------------- 验证

def walk_forward(series, events_series, horizon=HORIZON):
    """
    expanding 窗口验证。返回:
      {"ensemble": {h: {"mae","rmse","mape","n","std"}}, "holt": {...}}
    稀疏小区:仅统计 is_actual=True 的月份。
    """
    feats = build_full_features(series, events_series)
    feats = feats[feats[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)
    X, y = feats[FEATURE_COLS].values, feats["price"].values
    actual = feats["is_actual"].values
    n = len(feats)
    cutoffs = sorted(set(list(range(24, n - horizon, WALK_STEP)) + [n - horizon]))
    cutoffs = [c for c in cutoffs if c > 0]

    ens = Ensemble()
    pairs = {"ensemble": {h: [] for h in range(1, horizon + 1)},
             "holt": {h: [] for h in range(1, horizon + 1)}}
    for c in cutoffs:
        ens.fit(X[:c], y[:c])
        pred_ens = ens.predict(X[c: c + horizon])
        try:
            pred_holt = holt_forecast(y[:c], horizon)
        except Exception:
            pred_holt = np.full(horizon, y[c - 1])
        for h in range(horizon):
            j = c + h
            if j >= n or not actual[j]:
                continue
            for k, pred in (("ensemble", pred_ens), ("holt", pred_holt)):
                pairs[k][h + 1].append((y[j], pred[h]))

    out = {}
    for k, ph in pairs.items():
        out[k] = {}
        for h, plist in ph.items():
            if not plist:
                continue
            a = np.array([x[0] for x in plist])
            b = np.array([x[1] for x in plist])
            e = a - b
            out[k][h] = {
                "mae": float(np.abs(e).mean()),
                "rmse": float(np.sqrt((e ** 2).mean())),
                "mape": float((np.abs(e) / a).mean() * 100),
                "n": int(len(plist)),
                "std": float(e.std(ddof=1)) if len(e) > 1 else float(np.abs(e).mean()),
            }
    return out


def summarize(stats):
    """把 walk-forward 分步统计汇总为一行。"""
    if not stats:
        return None
    return {
        "MAE": np.mean([v["mae"] for v in stats.values()]),
        "RMSE": np.mean([v["rmse"] for v in stats.values()]),
        "MAPE%": np.mean([v["mape"] for v in stats.values()]),
        "验证月数": sum(v["n"] for v in stats.values()),
    }


# ---------------------------------------------------------------- 多轮训练 + 递归预测

def _row_price_features(comb, pos):
    """
    单行特征计算(与 build_price_features 全表口径完全一致,仅用 ≤t-1 的信息):
    供递归预测逐行使用,避免每填一个月就全表重算(66月×30轮下的性能关键)。
    边界口径:数据不足 lag 个月时,滞后价用序列首价近似 / 收益置 0(与 features 侧 fillna 一致)。
    """
    p = comb["price"].values
    f = {}

    def pv(i):
        # 越界处理:左边界(i<0)用首价近似,右边界(i>=len)用 NaN(未来尚未生成)
        if i < 0:
            return p[0] if len(p) else np.nan
        return p[i] if i < len(p) else np.nan

    for lag in (1, 3, 6, 12):
        f[f"p_lag{lag}"] = pv(pos - lag)
        if pos - lag - 1 >= 0 and p[pos - lag - 1]:
            f[f"r_lag{lag}"] = p[pos - lag] / p[pos - lag - 1] - 1
        else:
            f[f"r_lag{lag}"] = 0.0 if pos - lag - 1 < 0 else np.nan
    seg6 = p[max(0, pos - 6):pos]
    seg12 = p[max(0, pos - 12):pos]
    f["ma6"] = seg6.mean() if len(seg6) else (p[0] if len(p) else np.nan)
    f["ma12"] = seg12.mean() if len(seg12) else (p[0] if len(p) else np.nan)
    seg13 = p[max(0, pos - 13):pos]
    rets = np.diff(seg13) / seg13[:-1] if len(seg13) >= 3 else np.array([])
    # 与 pandas rolling.std 一致:ddof=1;样本不足时波动置 0(与 features 侧 fillna(0) 一致)
    f["vol12"] = rets.std(ddof=1) if len(rets) >= 2 else 0.0
    if pos > 0:
        peak = float(np.max(p[:pos]))
        f["dd_ratio"] = p[pos - 1] / peak if peak else 1.0
        cands = np.where(p[:pos] >= peak * (1 - 1e-9))[0]
        f["months_since_peak"] = float((pos - 1) - (cands[-1] if len(cands) else 0))
    else:
        f["dd_ratio"] = 1.0
        f["months_since_peak"] = 0.0
    f["t_idx"] = float(pos)
    return f


def recursive_forecast(series, events_series, scenario, n, rounds=N_ROUNDS,
                       seed=2026, on_round=None, weights=None):
    """
    多轮(bootstrap)× 递归多步预测:
      每轮:有放回抽样训练月 → 集成模型拟合 → 逐月递归预测;
            每月用已填值(历史+此前预测)计算该月特征(滞后/MA/波动率/回撤,
            事件特征按情景事件流固定),口径与训练一致、无泄漏。
    返回 DataFrame:date + 每轮一列预测(用于汇总 P10/P50/P90)。
    on_round: 可选回调 on_round(已完成轮数, 总轮数),用于进度显示。
    """
    from features import future_event_flow

    last = series["date"].max()
    future_dates = pd.date_range(last + pd.offsets.MonthEnd(1), periods=n, freq="ME")
    comb = pd.concat([
        series[["date", "price", "is_actual"]],
        pd.DataFrame({"date": future_dates, "price": np.nan, "is_actual": False}),
    ], ignore_index=True)
    # 历史 + 未来事件一次性拼好;事件特征在递归中固定
    events_full = pd.concat([events_series, future_event_flow(events_series, scenario, n)])
    comb = build_full_features(comb, events_full)

    # 训练行:价格已知 且 特征完整(前12行滞后特征缺失,不参与训练)
    valid = comb["price"].notna() & comb[FEATURE_COLS].notna().all(axis=1)
    known_idx = comb.index[valid].values
    fut_idx = comb.index[~comb["price"].notna()].values
    pred_cols = {}
    rng = np.random.default_rng(seed)

    for r in range(rounds):
        X_all = comb[FEATURE_COLS].values
        y_all = comb["price"].values
        boot = rng.choice(known_idx, size=len(known_idx), replace=True)
        ens = Ensemble(weights=weights)
        ens.fit(X_all[boot], y_all[boot])
        for pos in fut_idx:
            # 仅重算本行特征(事件列不变,直接沿用)
            row = _row_price_features(comb, pos)
            for k, v in row.items():
                comb.loc[pos, k] = v
            X_next = comb.loc[pos, FEATURE_COLS].values.reshape(1, -1)
            pred = ens.predict(X_next)[0]
            comb.loc[pos, "price"] = pred
        pred_cols[r] = comb.loc[fut_idx, "price"].values
        comb.loc[fut_idx, "price"] = np.nan   # 清空未来价格,进入下一轮
        if on_round is not None:
            on_round(r + 1, rounds)

    out = pd.DataFrame(pred_cols)
    out.insert(0, "date", future_dates.strftime("%Y-%m"))
    return out


def estimate_event_beta(series, events_series):
    """
    估计"事件→价格"弹性 β:对月度对数收益 Δln(p) 回归于 ev_net12(近12个月净事件方向)。
    正 β = 政策宽松(正向事件)伴随价格上涨;系数来自历史相关性,方向与量级由数据决定。
    返回 (beta12, 样本数);若事件稀疏返回 0。
    """
    from sklearn.linear_model import Ridge
    df = build_full_features(series, events_series)
    df = df[df[FEATURE_COLS].notna().all(axis=1)].reset_index(drop=True)
    y = np.log(df["price"]).diff().values[1:]
    X = df["ev_net12"].values[1:].reshape(-1, 1)
    if len(X) < 12 or np.all(X == 0):
        return 0.0, len(X)
    m = Ridge(alpha=10.0).fit(X, y)
    return float(m.coef_[0]), len(X)


def summarize_rounds(round_df):
    """多轮预测 → 每行(P10, P50, P90, mean)。"""
    vals = round_df.drop(columns=["date"]).values
    return pd.DataFrame({
        "date": round_df["date"],
        "P10": np.percentile(vals, 10, axis=1),
        "P50": np.percentile(vals, 50, axis=1),
        "P90": np.percentile(vals, 90, axis=1),
        "mean": vals.mean(axis=1),
    })
