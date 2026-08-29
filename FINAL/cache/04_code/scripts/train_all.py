# -*- coding: utf-8 -*-
"""
train_all.py — 联合训练:用 数据快照/ALL 下所有小区的数据训练一个面板(pooled)模型。

设计:
  1. 每个小区:二手房综合均价 除以 该小区训练期均值 → 标准化价格(跨小区可比,
     模型学"涨跌形态"而非"价格水平");小区标识 complex_id 作为特征。
  2. 特征 = 标准价格形态(滞后/动量/均线/波动率/回撤)+ 时间 + 事件(共享)+ complex_id。
  3. 30 轮 bootstrap:有放回抽样(按小区×月份行)训练 Ridge+RF+GBR 集成,
     保存 30 个模型到 CHECK/models/ensemble_all.joblib(预测阶段直接加载,无需重训)。
  4. 进度窗口 + 控制台进度。

运行: python scripts/train_all.py  (或双击 训练.bat)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import joblib

from load_data import load_complex, WORKBOOK_DIR
from features import (build_events, load_extra_events, build_price_features, attach_events,
                      FEATURE_COLS, load_macro, build_macro_frame, MACRO_COLS,
                      load_attrs, ATTRIBUTE_COLS, ATTR_NUM)
from models import Ensemble

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DIR = os.path.join(ROOT, "data", "train")
MODEL_DIR = os.path.join(ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_FILE = os.path.join(MODEL_DIR, "ensemble_all.joblib")
TRAIN_CUTOFF = pd.Timestamp("2024-08-31")   # 测试用小区(北控/帝泊湾)按此截断;ALL 小区全量参与
MIN_HISTORY = 12                            # 少于 12 个月历史的小区跳过
# 注意:12-23 个月的短样本以"短历史"身份参与(滞后12月等特征缺失的行会被
# build_pooled 自动过滤,不产生坏样本);24 个月以上才启用全部特征。

# ---- 最高强度训练配置 ----
# 训练轮数(有放回 bootstrap)× 树数 × 深度 × 学习率:
#   100 轮 × 500 棵树(RF/GBR 各 500),RF 深 5、GBR 深 4、lr 0.02、叶节点最小 3 样本。
# 防过拟合约束仍保留:bootstrap 有放回抽样 + 叶节点最小样本 + 深度上限 + 校准自动开关。
# ⚠ 代价:训练约 10-15 分钟,模型文件约 2-3GB,预测约 3-5 分钟。
# 如觉过慢/过大,可下调 TRAIN_ROUNDS(如 60)或 N_TREES(如 300)。
TRAIN_ROUNDS = 100
N_TREES = 500
HIGH_CFG = dict(n_estimators=N_TREES, rf_depth=5, gbr_depth=4, lr=0.02, min_leaf=3)
HALF_LIFE = 24   # 时间衰减半衰期(月):24 个月前的样本权重约为近期的一半


# ---- CPU 拉满并行配置 ----
# 之前 CPU 利用率低的原因:① GBR 串行 boosting(每棵树依赖前一棵残差,只能单核);
# ② RF 没设 n_jobs,默认单核。现在:
#   100 轮 bootstrap 互相独立 → 按逻辑核数并行进程(ProcessPoolExecutor);
#   每进程内 RF n_jobs=1(进程数=核数,再开线程会超订抖动;GBR 无并行参数)。
# 8 逻辑核 → 8 轮同时训练,总耗时 ≈ 原来的 1/8(27 分钟 → ~4 分钟)。
N_JOBS = max(1, os.cpu_count() or 1)

_GLOBAL = {}


def _init_worker(Xv, yv, wv):
    """worker 进程启动时一次性接收训练数据(spawn 模式下避免每轮重复传输大数组)。"""
    _GLOBAL["Xv"], _GLOBAL["yv"], _GLOBAL["wv"] = Xv, yv, wv


def _train_one_round(r, cfg):
    """单轮 bootstrap 训练(worker 进程内执行)。种子独立:20260000 + r,结果可复现。"""
    Xv, yv, wv = _GLOBAL["Xv"], _GLOBAL["yv"], _GLOBAL["wv"]
    rng = np.random.default_rng(20260000 + r)
    idx = rng.choice(len(Xv), size=len(Xv), replace=True)
    ens = Ensemble(n_jobs=1, **cfg)
    ens.fit(Xv[idx], yv[idx], sample_weight=wv[idx])
    return ens


def train_parallel(Xv, yv, wv, rounds, cfg, on_done=None, n_jobs=None):
    """
    轮间并行训练:bootstrap 每轮独立,按核数并行(CPU 拉满)。
    返回按轮次排序的模型列表(结果与串行训练逐轮对应)。
    on_done(done, total): 每完成一轮在主进程回调(进度条更新)。
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    if n_jobs is None:
        n_jobs = min(N_JOBS, rounds)
    results = [None] * rounds
    done = [0]
    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=_init_worker, initargs=(Xv, yv, wv)) as ex:
        futs = {ex.submit(_train_one_round, r, cfg): r for r in range(rounds)}
        for fut in as_completed(futs):
            r = futs[fut]
            results[r] = fut.result()
            done[0] += 1
            if on_done is not None:
                on_done(done[0], rounds)
    return results


def _make_progress():
    if os.environ.get("BK_NO_GUI"):
        return None
    try:
        from progress_gui import ProgressGUI
        return ProgressGUI(title="联合训练运行中")
    except Exception as e:
        print(f"[提示] 进度窗口创建失败({e}),改用控制台输出")
        return None


def load_all_complexes(include_targets=False):
    """
    扫描 data/train 目录解析每个小区(训练集)。
    include_targets 默认 False:训练集与测试集(data/test,北控/帝泊湾)物理隔离,
    测试小区绝不参与联合模型训练(严格测试集隔离,评估更诚实)。
    如特殊需要可传 True 纳入(会截断到 ≤2024.08 参与训练)。
    """
    # 递归扫描 ALL 全部子目录(01_deep_fall/ 02_v_rebound/ … 分类整理后的结构)
    paths = sorted(os.path.join(root, f) for root, _, fs in os.walk(ALL_DIR)
                   for f in fs if f.lower().endswith((".xlsx", ".xlsm"))
                   and not f.startswith("~$"))
    out = []
    for p in paths:
        try:
            c = load_complex(p)
            out.append(c)
        except Exception as e:
            print(f"  ⚠ 解析失败跳过: {os.path.basename(p)} ({e})")
    if include_targets:
        for f in sorted(os.listdir(WORKBOOK_DIR)):
            if f.lower().endswith((".xlsx", ".xlsm")) and not f.startswith("~$") \
                    and "宏观" not in f:   # 宏观数据表不是小区,跳过
                try:
                    c = load_complex(os.path.join(WORKBOOK_DIR, f))
                    # 目标小区截断:只用 ≤2024.08 参与训练(测试留出)
                    c["old"] = c["old"][c["old"]["date"] <= TRAIN_CUTOFF]
                    out.append(c)
                except Exception as e:
                    print(f"  ⚠ 解析失败跳过: {f} ({e})")
    return out


def build_pooled(complexes, events, macro=None, attrs=None):
    """
    构造面板特征矩阵(每行 = 某小区某月):
      X: 标准价格形态 + 时间 + 事件 + 宏观(lpr/全市均价) + 小区属性 + complex_id
      y: 标准化价格(price / 小区训练期均值)
      w: 时间衰减样本权重(近期样本权重更高,半衰期 HALF_LIFE 个月)——让模型更贴合
         近期行情(regime 切换时改善外推),同时不丢弃早期信息。
    macro: load_macro() 的结果;attrs: load_attrs() 的结果;None/空 → 自动退化。
    属性特征(导师建议):学区/楼龄/绿化率/容积率/距地铁/城区——让新小区能借用
    "相似小区"的知识预测(泛化),而非依赖 complex_id 外推。
    返回 (X, y, w, meta, x_cols, attr_info)。attr_info: {scale, medians} 供推理填充。
    """
    has_macro = bool(macro)
    has_attrs = attrs is not None and len(attrs) > 0
    x_cols = FEATURE_COLS + (MACRO_COLS if has_macro else []) \
        + (ATTRIBUTE_COLS if has_attrs else []) + ["complex_id"]
    # 属性数值列按全局均值缩放(Ridge 对尺度敏感:楼龄 0-50 vs 距地铁 0-20)
    attr_info = None
    if has_attrs:
        attr_scale = attrs[ATTR_NUM].mean().replace(0, np.nan).fillna(1.0)
        attr_medians = attrs.median().fillna(0.0)
        attr_info = {"scale": attr_scale.to_dict(), "medians": attr_medians.to_dict()}
    X_list, y_list, w_list, meta_rows = [], [], [], []
    for cid, c in enumerate(complexes):
        old = c["old"]
        if len(old) < MIN_HISTORY:
            print(f"  跳过 {c['name']}(历史 {len(old)} 个月 < {MIN_HISTORY})")
            continue
        mean_price = float(old["price"].mean())
        s = old[["date", "price", "is_actual"]].copy()
        s["price"] = s["price"] / mean_price          # 标准化
        s["complex_id"] = cid
        df = attach_events(build_price_features(s), events)
        if has_macro:
            df[MACRO_COLS] = build_macro_frame(macro, df["date"]).values
        if has_attrs:
            if c["name"] in attrs.index:
                row = attrs.loc[c["name"]].fillna(attr_medians)
            else:
                row = attr_medians                    # 无属性行的小区用全局中位数
            arow = row.copy()
            arow[ATTR_NUM] = arow[ATTR_NUM] / attr_scale
            df[ATTRIBUTE_COLS] = arow.values
        df = df[df[x_cols].notna().all(axis=1)].reset_index(drop=True)
        # 时间衰减权重:第 i 行(时间升序)权重 = exp(-(n-1-i)/HALF_LIFE),越近权重越高
        n = len(df)
        df["_w"] = np.exp(-(n - 1 - np.arange(n)) / HALF_LIFE)
        X_list.append(df[x_cols])
        y_list.append(df["price"])
        w_list.append(df["_w"])
        meta_rows.append({"name": c["name"], "mean": mean_price,
                          "n": len(df), "id": cid})
    X = pd.concat(X_list, ignore_index=True)
    y = pd.concat(y_list, ignore_index=True)
    w = pd.concat(w_list, ignore_index=True)
    return X, y, w, meta_rows, x_cols, attr_info


def main():
    """独立运行入口:进度窗口在子线程,主线程刷新(窗口不"未响应")。"""
    from progress_gui import run_with_progress
    run_with_progress(main_impl, _make_progress, title="联合训练运行中")


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

    t0 = time.time()
    report("扫描 ALL 小区数据", 2)
    complexes = load_all_complexes()
    report(f"解析完成:{len(complexes)} 个小区", 6, f"{[c['name'] for c in complexes][:5]} …")

    # 事件:ALL 小区事件年表合并 + 外部事件
    all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                       ignore_index=True)
    events = build_events(all_ev, load_extra_events())
    report(f"事件构建:{len(events)} 条", 8)

    report("构建面板特征(标准化价格 + 小区标识 + 宏观 + 属性 + 时间衰减权重)", 10)
    macro = load_macro()
    attrs = load_attrs()
    X, y, w, meta, x_cols, attr_info = build_pooled(complexes, events, macro, attrs)
    if len(X) < 500:
        print(f"⚠ 面板样本仅 {len(X)} 行,联合训练效果有限")
    report(f"面板样本 {len(X)} 行 × {X.shape[1]} 特征,{len(meta)} 个小区"
           f"(近期样本加权,半衰期 {HALF_LIFE} 个月)", 12,
           f"特征列: {x_cols}")

    # 最高强度:100 轮 bootstrap 训练集成模型(带时间衰减样本权重)
    # 轮间并行:8 逻辑核 → 8 轮同时训练,CPU 拉满(GBR 串行部分由多进程分摊)。
    Xv, yv, wv = X.values, y.values, w.values
    report(f"轮间并行训练:{N_JOBS} 进程 × {TRAIN_ROUNDS} 轮(RF 500 树/GBR 早停)", 12)
    models = train_parallel(Xv, yv, wv, TRAIN_ROUNDS, HIGH_CFG,
                            on_done=lambda d, t: report(
                                f"训练中:{d}/{t} 轮(并行)", 12 + 82 * d / t, None))

    # 保存模型 + meta
    joblib.dump({"models": models, "meta": meta, "feature_cols": x_cols,
                 "has_macro": bool(macro), "attr_info": attr_info,
                 "n_rounds": TRAIN_ROUNDS, "cfg": HIGH_CFG,
                 "trained_at": time.strftime("%Y-%m-%d %H:%M")},
                MODEL_FILE)
    report("训练完成,模型已保存", 96, f"模型 → {MODEL_FILE}")
    print(f"\n训练完成: {len(complexes)} 个小区,{len(X)} 行样本,{TRAIN_ROUNDS} 个集成模型"
          f"({HIGH_CFG}),耗时 {time.time() - t0:.0f} 秒")

    # 训练完成后自动进入预测
    print("\n→ 开始预测(对原两个小区做 2024.09 起测试预测)…\n")
    from predict import main_impl as predict_main
    predict_main(pw=pw)

    if pw is not None:
        pw.done([f"联合模型 → {MODEL_FILE}",
                 "预测已完成:打分表 + 对比图已输出"])


if __name__ == "__main__":
    main()
