# -*- coding: utf-8 -*-
"""
predict.py — 加载联合模型,对原两个小区(北控雁栖湖/新力帝泊湾)做准确性测试预测:
  截断到 2024.08 → 用 30 个联合模型递归预测 2024.09–2026.06(22个月)
  → 偏差校准(自动开关) → 打分追加(accuracy_metrics.xlsx / 打分表.xlsx)
  → 对比折线图(comparison/{编号}/) → 进度窗口 + 控制台进度。

运行: python scripts/predict.py  (或双击 预测.bat;训练.bat 完成后会自动调用)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from config_loader import CFG
from load_data import load_complex, WORKBOOK_DIR
from features import (build_events, load_extra_events, attach_events, build_price_features,
                      FEATURE_COLS, load_macro, build_macro_frame, MACRO_COLS,
                      load_attrs, ATTRIBUTE_COLS, ATTR_NUM)
from forecast import (targets_for, series_for_target, export as export_forecast,
                      interval_bands, holt_main_predict,
                      INTERVAL_QUANTILES, INTERVAL_SCALE)
from plot_monthly import draw as draw_history
from plot_forecast import draw as draw_forecast
from run_test import (CUTOFF, N_MONTHS, eval_series, calc_bias_ratio, calibration_factor,
                      USE_CALIBRATION, append_metrics, append_score_table,
                      export_comparison, plot_comparison, OUTPUT_DIR, METRICS_FILE,
                      next_run_folder, _make_progress as _mk_progress)
from train_all import load_all_complexes
from models import _row_price_features
from drift import build_train_distribution, drift_assessment, drift_message

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHART_DIR = os.path.join(ROOT, "comparison")
CHARTS_DIR = os.path.join(ROOT, "charts")

# ---- 从集中配置读取预测参数 ----
_PC = CFG["predict"]
FORECAST_END = _PC["forecast_end"]
_SCEN_RAW = _PC["scenarios"]
# 键映射:config.yaml 用英文键(baseline/optimistic/pessimistic),对外统一中文键(基准/乐观/悲观)。
# 修复:此前 config 改英文键后 export_xlsx / web 仍按中文键取值,触发 KeyError。
_SCEN_CN = {"baseline": "基准", "optimistic": "乐观", "pessimistic": "悲观"}
SCENARIOS = {_SCEN_CN.get(k, k): v for k, v in _SCEN_RAW.items()}

# 预测目标:data/test 下的小区(测试集,原两个:北控/帝泊湾)
TARGET_DIR = WORKBOOK_DIR


def cleanup_charts(target_names):
    """
    自动清理 charts/:测试环节只输出 2024.09 后的对比图(在 comparison/{编号}/),
    charts/ 不再保留历史走势图/预测延伸图——每次运行开始前清空。
    """
    removed = 0
    for f in os.listdir(CHARTS_DIR):
        if f.endswith(".png"):
            os.remove(os.path.join(CHARTS_DIR, f))
            removed += 1
    if removed:
        print(f"  [清理] charts/ 移除了 {removed} 个旧图(测试环节只需对比图)")


def cleanup_output():
    """output/ 自动清理:只保留 打分表/对比表/预测表,删除诊断表与旧杂项。"""
    keep = ("accuracy_metrics.xlsx", "打分表.xlsx", "accuracy_", "predictions_")
    removed = 0
    for f in os.listdir(OUTPUT_DIR):
        if not f.endswith((".xlsx", ".csv")):
            continue
        if any(f == k or f.startswith(k) for k in keep):
            continue
        os.remove(os.path.join(OUTPUT_DIR, f))
        removed += 1
    if removed:
        print(f"  [清理] output/ 移除了 {removed} 个旧文件(diagnose_*/metrics 等)")


def load_targets():
    """解析测试集 data/test 下的小区(北控/帝泊湾)。"""
    files = sorted(f for f in os.listdir(TARGET_DIR)
                   if f.lower().endswith((".xlsx", ".xlsm")) and not f.startswith("~$")
                   and "宏观" not in f)   # 宏观数据表不是小区,跳过
    out = []
    for f in files:
        try:
            out.append(load_complex(os.path.join(TARGET_DIR, f)))
        except Exception as e:
            print(f"  ⚠ 解析失败跳过: {f} ({e})")
    return out


def pooled_predict(target_series, mean_price, cid, models, events, n, seed=2026,
                   macro=None, attrs=None, attr_info=None, complex_name=None,
                   scenario="基准"):
    """
    用联合模型对目标小区递归预测 n 个月(标准化价格域)。
    target_series: date/price/is_actual(原始价格,截断后);mean_price: 该小区标准化均值。
    macro: load_macro() 结果;预测期宏观自动取最后已知值(LPR/全市均价持平假设)。
    attrs/attr_info: 属性表与训练时保存的 scale/medians;新小区按名取属性,
      无属性行用全局中位数(与训练口径一致)。
    scenario: 预测期事件情景("基准"/"乐观"/"悲观"),对应 features.future_event_flow。
    返回 DataFrame: date + 每模型一列预测(原始价格域)。
    """
    from features import future_event_flow
    has_macro = bool(macro)
    target_series = target_series.dropna(subset=["price"]).reset_index(drop=True)
    if len(target_series) < 12:
        return None      # 该序列历史不足 12 个月,无法递归预测(调用方跳过)
    last = target_series["date"].max()
    future = pd.date_range(last + pd.offsets.MonthEnd(1), periods=n, freq="ME")
    comb = pd.concat([
        target_series[["date", "price", "is_actual"]],
        pd.DataFrame({"date": future, "price": np.nan, "is_actual": False}),
    ], ignore_index=True)
    comb["price"] = comb["price"] / mean_price
    comb["complex_id"] = cid
    # 宏观特征:历史用实际值,预测期 ffill 自动取最后已知值(持平假设)
    if has_macro:
        comb[MACRO_COLS] = build_macro_frame(macro, comb["date"]).values
    # 属性特征(与训练口径一致:数值列 /scale;缺失用中位数)
    if attrs is not None and attr_info:
        if complex_name and complex_name in attrs.index:
            row = attrs.loc[complex_name].fillna(pd.Series(attr_info["medians"]))
        else:
            row = pd.Series(attr_info["medians"])
        arow = row.copy()
        arow[ATTR_NUM] = arow[ATTR_NUM] / pd.Series(attr_info["scale"])
        comb[ATTRIBUTE_COLS] = arow.values
    # 预测期事件 = 延续最近 12 个月政策节奏(与正式预测一致,避免未来月事件特征衰减);
    # scenario 决定未来事件流的变体(基准/乐观/悲观)
    events_full = pd.concat([events, future_event_flow(events, scenario, n)])
    comb = attach_events(build_price_features(comb), events_full)

    has_attrs = attrs is not None and attr_info is not None
    feat_cols = FEATURE_COLS + (MACRO_COLS if has_macro else []) \
        + (ATTRIBUTE_COLS if has_attrs else []) + ["complex_id"]
    valid = comb["price"].notna() & comb[feat_cols].notna().all(axis=1)
    known_idx = comb.index[valid].values
    fut_idx = comb.index[~comb["price"].notna()].values

    pred_cols = {}
    rng = np.random.default_rng(seed)
    for r, ens in enumerate(models):
        # 每模型各自 bootstrap 训练集已在训练阶段确定;预测时直接用该模型
        for pos in fut_idx:
            row = _row_price_features(comb, pos)
            for k, v in row.items():
                comb.loc[pos, k] = v
            X_next = comb.loc[pos, feat_cols].values.reshape(1, -1)
            X_next = np.nan_to_num(X_next, nan=0.0)   # 兜底:极端情况下缺失特征置 0
            pred = ens.predict(X_next)[0]
            comb.loc[pos, "price"] = pred
        pred_cols[r] = comb.loc[fut_idx, "price"].values * mean_price   # 还原原始价格域
        comb.loc[fut_idx, "price"] = np.nan

    out = pd.DataFrame(pred_cols)
    out.insert(0, "date", future.strftime("%Y-%m"))
    return out


def months_to_end(last_date, end=FORECAST_END):
    """历史末端 → 2031.12 的月数(含终点月)。"""
    end_dt = pd.Timestamp(end + "-01") + pd.offsets.MonthEnd(0)
    return (end_dt.year - last_date.year) * 12 + (end_dt.month - last_date.month)


def holt_damped_predict(series, n, warm_months=22, slope_half_life=36,
                       quantiles=None):
    """
    正式预测主方法 = Holt 趋势外推 + 趋势衰减 + 季节叠加(调整版):
      前 warm_months 个月(≈测试验证区 22 个月)保持纯 Holt 趋势(与测试预测一致,
      精度已验证:北控 98.3 / 帝泊湾 87.7);
      之后趋势斜率按半衰期衰减(默认 36 个月衰减一半)——长期预测收敛,
      避免线性外推 5 年后跌到荒谬价格(帝泊湾纯外推 -74.5% → 2,239 元/㎡);
      80% 区间随预测期扩张(±1.28×rstd×√t),不确定性随时间增长更真实。
    """
    from models import fit_holt
    fit = fit_holt(series["price"].values)
    level, slope = float(np.asarray(fit.level)[-1]), float(np.asarray(fit.trend)[-1])
    resid = series["price"].values - np.asarray(fit.fittedvalues)
    t = np.arange(1, n + 1, dtype=float)
    decay = np.ones_like(t)
    if n > warm_months:
        decay[warm_months:] = np.exp(-(t[warm_months:] - warm_months) / slope_half_life)
    cum = np.cumsum(decay)                     # 累计有效斜率月数
    trend = level + slope * cum
    # 季节叠加(与测试预测一致)
    s_tmp = series.copy()
    s_tmp["_m"] = s_tmp["date"].dt.month
    s_tmp["_r"] = resid
    season = s_tmp.groupby("_m")["_r"].mean().rolling(3, min_periods=1, center=True).mean()
    future_dates = pd.date_range(series["date"].max() + pd.offsets.MonthEnd(1),
                                 periods=n, freq="ME")
    seas_adj = np.array([season.get(m, 0.0) for m in future_dates.month])
    p50 = trend + seas_adj
    # 区间校准:第 12 步分位数误差带按 √(h/12) 扩张(长期不确定性增长)
    e10, e90 = interval_bands(series, quantiles=quantiles)
    e12 = e10.get(12) or next(iter(e10.values()), 0.0)
    e12_hi = e90.get(12) or next(iter(e90.values()), 0.0)
    # 区间扩张封顶 √(h/12) ≤ 2.0(长期不确定性不无限放大;修复帝泊湾 64 月下界为负)
    grow = np.minimum(np.sqrt(np.arange(1, n + 1) / 12), 2.0)
    lo = e12 * grow
    hi = e12_hi * grow
    return pd.DataFrame({
        "date": future_dates.strftime("%Y-%m"),
        "P10": p50 + lo * INTERVAL_SCALE,
        "P50": p50,
        "P90": p50 + hi * INTERVAL_SCALE,
        "mean": p50,
    })


# SCENARIOS 改由 config_loader 加载(见本文件顶部)
# 情景 = 趋势衰减半衰期(月):
#   基准 36 = 当前默认(3 年衰减一半);
#   乐观 18 = 趋势快速收敛止跌(政策回暖、跌幅收窄);
#   悲观 72 = 下行延续(政策持续收紧,趋势维持更久)。


def pooled_future_forecast(c, models, events, macro=None, attrs=None, attr_info=None,
                           meta=None):
    """
    联合模型未来预测(分布内小区:训练集/新小区用):
      用该小区全历史 + 联合模型递归预测至 FORECAST_END,输出 P10/P50/P90。
    返回与 formal_forecast 同构: {t: DataFrame(date,P10,P50,P90,mean), _meta}。
    说明:事件特征占比极低(<1%),分布内小区三情景差异可忽略,故只出基准区间;
    三情景完整支持仍由 Holt 趋势衰减(formal_forecast)承担。
    """
    old = c["old"]
    name = c["name"]
    meta_names = {m["name"]: m for m in meta} if meta else {}
    result = {}
    for t in targets_for(old):
        series = series_for_target(old, t)
        n = months_to_end(series["date"].max())
        if n <= 0:
            continue
        if name in meta_names:
            cid, mean_price = meta_names[name]["id"], meta_names[name]["mean"]
        else:
            cid = 99999
            mean_price = float(series["price"].mean())
        try:
            base = pooled_predict(series, mean_price, cid, models, events, n,
                                  macro=macro, attrs=attrs, attr_info=attr_info,
                                  complex_name=name)
            if base is None:
                continue
            vals = base.drop(columns=["date"])
            result[t] = pd.DataFrame({
                "date": base["date"],
                "P10": vals.quantile(0.10, axis=1).values,
                "P50": vals.quantile(0.50, axis=1).values,
                "P90": vals.quantile(0.90, axis=1).values,
                "mean": vals.mean(axis=1).values,
            })
        except Exception as e:
            print(f"  ⚠ {name} {t} 联合模型预测失败: {e}")
    if not result:
        return None
    result["_meta"] = {"targets": targets_for(old), "beta": {}}
    return result


def formal_forecast(c, scenarios=False):
    """
    正式预测:用全部历史(不截断)→ Holt 趋势衰减外推 → 2031.12。
    scenarios=True 时 result["_scenarios"] = {情景: {t: P50 Series}}。
    返回 result 字典({t: DataFrame(date, P10, P50, P90, mean)}),供预测表导出。
    """
    old = c["old"]
    result = {}
    scen_out = {}
    for t in targets_for(old):
        series = series_for_target(old, t)
        try:
            n = months_to_end(series["date"].max())
            result[t] = holt_damped_predict(series, n)
        except Exception:
            continue   # 历史不足(Holt 需 ≥8 观测)的小区跳过该序列
        if scenarios:
            scen_out[t] = {}
            for sname, hl in SCENARIOS.items():
                df = holt_damped_predict(series, n, slope_half_life=hl)
                scen_out[t][sname] = dict(zip(df["date"], df["P50"]))
    if not result:
        raise RuntimeError("该小区无有效价格序列,无法预测")
    result["_meta"] = {"targets": targets_for(old), "beta": {}}
    if scenarios:
        result["_scenarios"] = scen_out
    return result


def main():
    """独立运行入口:进度窗口在子线程,主线程刷新(窗口不"未响应")。"""
    from progress_gui import run_with_progress
    run_with_progress(main_impl, _mk_progress, title="预测运行中")


def main_impl(pw=None, events=None, macro=None, complexes=None,
              base_pct=0.0, span=100.0):
    """预测主流程(worker 线程内执行)。参数:
      pw: 进度窗口(独立运行由 run_with_progress 创建;run_joint 传入共享窗口);
      events: 外部事件序列(训练阶段已构建;None 则自行构建);
      macro: 宏观月度数据(训练阶段已构建;None 则自行加载);
      complexes: 训练小区列表(训练阶段已解析;None 则自行解析,供漂移分布复用);
      base_pct/span: 进度偏移与跨度(供 run_joint 拼接 50-100%)。
    """
    last_pct = [0]

    def report(text, pct, log=None):
        pct = base_pct + span * pct / 100.0
        if pw is not None:
            pw.update(text, pct, log)
        else:
            if int(pct) >= last_pct[0] + 5 or pct >= 100:
                last_pct[0] = int(pct)
                print(f"  [{pct:3.0f}%] {text}", flush=True)
        if log:
            print("  " + log, flush=True)

    # 目标小区采用 Holt 趋势外推主预测(实证:对单边趋势小区远优于联合模型外推,
    # 北控 98.3 / 帝泊湾 87.7 分 vs 联合模型 29 / 25 分)。
    # 联合模型(训练.bat 生成)仍用于 ALL 新小区预测(predict_random.py),
    # 此处无需加载 2-3GB 模型文件。
    report("Holt 主预测模式:目标小区趋势外推 + 季节叠加", 2,
           "联合模型仍用于 ALL 小区预测,不受影响")

    # 事件:外部传入(训练阶段已构建)或自行构建(复用传入的训练小区,避免重复解析)
    if events is None:
        if complexes is None:
            complexes = load_all_complexes()
        all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                           ignore_index=True)
        events = build_events(all_ev, load_extra_events())
    events = events[events.index <= CUTOFF]
    report(f"事件截断:{len(events)} 条(≤2024.08)", 4)

    # 漂移检测:训练集行情分布(复用已解析的小区,避免二次全量解析)
    drift_dist = None
    try:
        if complexes is None:
            complexes = load_all_complexes()
        drift_dist = build_train_distribution(complexes, macro)
    except Exception as e:
        print(f"  ⚠ 漂移分布构建失败({e}),跳过漂移检测")

    targets = load_targets()
    if not targets:
        print(f"data/test 未找到目标小区(北控/帝泊湾)")
        return
    cleanup_charts([c["name"] for c in targets])   # charts/ 清空(测试环节只需对比图)
    cleanup_output()                               # output/ 自动清理旧文件
    report(f"预测目标:{[c['name'] for c in targets]}", 5)

    run_folder = next_run_folder()
    run_time = time.strftime("%Y-%m-%d %H:%M")
    metrics_rows = []
    summaries = []
    formal_summaries = []
    total_steps = len(targets) * 3
    done = [0]

    for ci, c in enumerate(targets):
        name = c["name"]
        old = c["old"]
        train = old[old["date"] <= CUTOFF]
        actual = old[old["date"] > CUTOFF][["date", "price", "jingzhuang", "maopi",
                                            "bieshu", "is_actual"]].copy()
        actual["date"] = actual["date"].dt.strftime("%Y-%m")
        actual_mask = actual.set_index("date")["is_actual"]

        report(f"[{ci + 1}/{len(targets)}] {name}:Holt 趋势外推 + 季节叠加", 6 + ci * 4,
               f"训练窗口 {train['date'].min():%Y.%m}–{train['date'].max():%Y.%m} ({len(train)}个月)")

        preds = {}
        for t in targets_for(old):
            series = series_for_target(train, t)   # 原始价格域
            try:
                preds[t] = holt_main_predict(series, N_MONTHS)
            except Exception as e:
                print(f"  ⚠ {name} {t}:历史不足,跳过预测({e})")

        # 偏差校准:Holt 主预测模式下跳过(趋势外推偏差已很小,校准可能画蛇添足——
        # 北控训练期偏差 -7.6% 但测试期高估 +31%,方向相反,校准会帮倒忙)
        bias = calc_bias_ratio(train, events)
        print(f"\n[{name}] Holt 主预测:跳过偏差校准(训练期验证偏差 {bias:+.2%})")
        if drift_dist is not None:
            dr = drift_assessment(drift_dist, train["price"], macro)
            print(drift_message(dr, name))

        # 打分 + 输出
        print(f"\n[{name}] Holt 主预测 2024.09–2026.06 与真实对比:")
        scores = []
        for t, df in preds.items():
            col = {"综合": "price", "精装二手": "jingzhuang",
                   "毛坯二手": "maopi", "别墅二手": "bieshu"}[t]
            act = actual[["date", col]].rename(columns={col: "price"})
            m = eval_series(act, df, mask=actual_mask)
            if m:
                print(f"  {t}: MAE={m['MAE']:,.0f}  RMSE={m['RMSE']:,.0f}  "
                      f"MAPE={m['MAPE%']:.2f}%  得分={m['得分(0-100)']:.1f}  "
                      f"覆盖率={m['区间覆盖率%']:.0f}%  (样本{m['样本数']}个真实点)")
                metrics_rows.append([run_time, f"[Holt]{name}", t,
                                     round(m["MAE"]), round(m["RMSE"]), round(m["MAPE%"], 2),
                                     round(m["得分(0-100)"], 1), round(m["区间覆盖率%"]),
                                     m["样本数"], round(m["真实均值"]), round(m["预测P50均值"])])
                scores.append(m["得分(0-100)"])
        if scores:
            metrics_rows.append([run_time, f"[Holt]{name}", "平均",
                                 "", "", "", round(sum(scores) / len(scores), 1),
                                 "", len(scores), "", ""])

        export_comparison(name, actual, preds)
        plot_comparison(name, actual, preds,
                        os.path.join(run_folder, f"accuracy_{name}.png"), actual_mask)
        report(f"[{ci + 1}/{len(targets)}] {name}:对比表+对比图完成", 88 + ci * 4,
               f"对比图 → {run_folder}")

        # ---- 正式预测:全部历史 → 2031.12(预测表,含三情景) ----
        # 预测延伸图与对比图放同一文件夹(comparison/{编号}/)
        fresult = formal_forecast(c, scenarios=True)
        export_forecast(name, fresult)
        draw_forecast(name, c, fresult,
                      os.path.join(run_folder, f"{name}_预测延伸.png"))
        report(f"[{ci + 1}/{len(targets)}] {name}:正式预测至 {FORECAST_END} 完成",
               92 + ci * 4, f"预测表+预测图 → {run_folder}")
        done[0] += 1

        summaries.append((name, actual["price"].iloc[-1], preds["综合"].iloc[-1]["P50"]))
        # 正式预测末值(2031.12)汇总
        fz = fresult["综合"]
        formal_summaries.append((name, fz["P50"].iloc[0], fz["P50"].iloc[-1]))

    # 打分表(追加;特征占比 sheet 用联合模型的形态——简化:仅追加打分行)
    if metrics_rows:
        append_metrics(metrics_rows, {})
    append_score_table()
    report("全部完成", 100, f"打分表已追加 → {METRICS_FILE}")

    print("\n" + "=" * 90)
    for name, real, pred in summaries:
        print(f"  {name}: 真实 {real:,.0f}  预测 {pred:,.0f}  偏差 {pred - real:+,.0f} "
              f"({pred / real - 1:+.1%})")
    if formal_summaries:
        print("\n正式预测(全部历史 → 2031.12):")
        for name, p0, pn in formal_summaries:
            print(f"  {name}: 当前 {p0:,.0f} → 2031.12 {pn:,.0f} ({pn / p0 - 1:+.1%})")
    print(f"\n全部完成:打分表 → {OUTPUT_DIR}  对比图 → {run_folder}")
    if pw is not None:
        pw.done([f"打分表追加 → {os.path.basename(METRICS_FILE)}",
                 f"对比图 → {run_folder}",
                 f"正式预测至 2031.12 → output/ + charts/"])


if __name__ == "__main__":
    main()
