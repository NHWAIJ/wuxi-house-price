# -*- coding: utf-8 -*-
"""
web/app.py — 网页演示应用(答辩用):
  选择小区 → 历史走势 + 正式预测(至2031.12,50%区间+三情景)+ 多截断点诊断 + 漂移提示。
  加载小区与预测均为后台线程执行,前端轮询真实进度(百分比 + 预计剩余秒数)。
启动: 双击 FINAL/web.bat(自动开浏览器 http://127.0.0.1:5000)
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import pandas as pd

from flask import Flask, jsonify, render_template, request

from load_data import load_complex, WORKBOOK_DIR
from features import load_macro
from predict import formal_forecast, FORECAST_END, SCENARIOS
from drift import build_train_distribution, drift_assessment
from target_diagnose import diagnose_one

app = Flask(__name__)

# 并发保护:预测/训练接口先同步置位再起线程,避免两个请求同时通过检查
PRED_LOCK = threading.Lock()
TRAIN_LOCK = threading.Lock()


@app.after_request
def no_cache(resp):
    """禁用浏览器缓存:保证网页更新后用户立即看到新版本(修复'没进度条'的缓存问题)。"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/health")
def api_health():
    """健康检查(Docker HEALTHCHECK / 运维探活)。"""
    return jsonify({"status": "ok", "loaded": bool(CACHE.get("loaded"))})

# ---------------- 小区加载(线程化,真实进度) ----------------
CACHE = {"loaded": False}
LOAD_STATE = {"running": False, "stage": "", "done": 0, "total": 0,
              "started_at": 0.0, "msg": ""}


def _load_job():
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        CACHE["loaded"] = False
        LOAD_STATE.update(running=True, done=0, total=0,
                          started_at=time.time(), msg="")
        TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "train")
        paths = sorted(os.path.join(root, f) for root, _, fs in os.walk(TRAIN_DIR)
                       for f in fs if f.lower().endswith((".xlsx", ".xlsm"))
                       and not f.startswith("~$"))
        test_paths = [os.path.join(WORKBOOK_DIR, f) for f in sorted(os.listdir(WORKBOOK_DIR))
                      if f.lower().endswith((".xlsx", ".xlsm")) and "宏观" not in f]
        all_paths = paths + test_paths
        n_cpu = os.cpu_count() or 4
        n_workers = min(8, max(2, n_cpu // 2))  # I/O 密集,半核数就够
        LOAD_STATE.update(total=len(all_paths) + 3,  # +宏观/属性/漂移
                          stage=f"并行解析 {len(all_paths)} 个小区 ({n_workers} 线程)")
        complex_map = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(load_complex, p): p for p in all_paths}
            for i, fut in enumerate(as_completed(futs)):
                p = futs[fut]
                try:
                    c = fut.result()
                    complex_map[p] = c
                except Exception:
                    pass
                if i % 10 == 0 or i == len(all_paths) - 1:
                    LOAD_STATE.update(done=i + 1,
                                      stage=f"解析小区 {i + 1}/{len(all_paths)} ({n_workers} 线程)")
        # 按原始顺序恢复
        complexes = [complex_map[p] for p in paths if p in complex_map]
        test = [complex_map[p] for p in test_paths if p in complex_map]
        LOAD_STATE.update(stage="构建宏观与漂移分布", done=len(all_paths))
        macro = load_macro()
        attrs = None
        try:
            from features import load_attrs
            attrs = load_attrs()
        except Exception:
            pass
        dist = build_train_distribution(complexes, macro)
        CACHE.update({"complexes": complexes, "test": test, "macro": macro,
                      "attrs": attrs, "dist": dist,
                      "loaded": True})
        LOAD_STATE.update(done=LOAD_STATE["total"], stage="完成",
                          msg=f"加载完成:{len(complexes)} 个训练小区")
    except Exception as e:
        import traceback
        LOAD_STATE.update(stage="失败", msg=str(e)[:200] + " | " + traceback.format_exc()[-200:])
    finally:
        LOAD_STATE["running"] = False


def ensure_loaded():
    """未加载则启动加载线程(不阻塞请求)。"""
    if not CACHE["loaded"] and not LOAD_STATE["running"]:
        threading.Thread(target=_load_job, daemon=True).start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/complexes")
def api_complexes():
    ensure_loaded()
    if CACHE["loaded"]:
        d = CACHE
        items = ([{"name": c["name"], "group": "测试集"} for c in d["test"]] +
                 [{"name": c["name"], "group": "训练集"} for c in d["complexes"]])
        return jsonify({"items": items, "ready": True})
    return jsonify({"items": [], "ready": False})


@app.route("/api/load_progress")
def api_load_progress():
    ensure_loaded()
    st = dict(LOAD_STATE)
    if st["started_at"] and st["done"] > 0 and st["total"] > 0:
        elapsed = time.time() - st["started_at"]
        st["eta"] = elapsed / st["done"] * (st["total"] - st["done"])
        st["elapsed"] = elapsed
    else:
        st["eta"] = 0.0
        st["elapsed"] = 0.0
    st["ready"] = bool(CACHE["loaded"])
    return jsonify(st)


# ---------------- 预测(异步,真实进度) ----------------
PRED_STATE = {"running": False, "stage": "", "pct": 0, "started_at": 0.0,
              "name": "", "msg": "", "result": None}


def _predict_job(name):
    try:
        # running 已在 api 层同步置位;这里只需更新阶段
        PRED_STATE.update(name=name, msg="", result=None,
                          stage="正在启动预测…", pct=1, started_at=time.time())
        # 1. 等待小区加载完成(若尚未加载)
        while not CACHE["loaded"]:
            if LOAD_STATE["stage"] == "失败":
                raise RuntimeError("小区数据加载失败")
            PRED_STATE.update(stage="等待小区数据加载…", pct=5)
            time.sleep(0.5)
        d = CACHE
        c = None
        for cc in d["test"] + d["complexes"]:
            if cc["name"] == name:
                c = cc
                break
        if c is None:
            raise RuntimeError("未找到小区:" + name)
        old = c["old"]

        # 2. 历史数据(10%)
        PRED_STATE.update(stage="整理历史数据…", pct=10)
        hist = []
        for _, r in old.iterrows():
            hist.append({
                "date": r["date"].strftime("%Y-%m"),
                "zonghe": None if pd.isna(r["zonghe"]) else round(float(r["zonghe"])),
                "jingzhuang": None if pd.isna(r["jingzhuang"]) else round(float(r["jingzhuang"])),
                "maopi": None if pd.isna(r["maopi"]) else round(float(r["maopi"])),
            })

        # 3. 漂移检测(15%)
        PRED_STATE.update(stage="分布漂移检测…", pct=15)
        dr = drift_assessment(d["dist"], old["price"], d["macro"])

        # 4. 多截断点诊断(15%-55%,4 个截断点)
        rows, summary, _ = diagnose_one(c, None)
        PRED_STATE.update(stage="多截断点诊断完成", pct=55)

        # 5. 正式预测(55%-90%,三情景)
        PRED_STATE.update(stage="正式预测(三情景)…", pct=70)
        try:
            fresult = formal_forecast(c, scenarios=True)
        except Exception as e:
            raise RuntimeError(f"正式预测失败: {e}")
        f = fresult.get("综合") or next(iter(fresult.values()), None)
        if f is None:
            raise RuntimeError("该小区无有效价格序列,无法预测")
        scen = fresult.get("_scenarios", {}).get("综合", {})
        pred = []
        for _, row in f.iterrows():
            pred.append({
                "date": row["date"],
                "p10": round(float(row["P10"])), "p50": round(float(row["P50"])),
                "p90": round(float(row["P90"])),
                "opt": round(float(scen["乐观"][row["date"]])),
                "pess": round(float(scen["悲观"][row["date"]])),
            })
        cur = hist[-1]["zonghe"] if hist else None
        last = pred[-1] if pred else None
        lpr = None
        if d["macro"] and "5年期LPR" in d["macro"]:
            lpr = float(d["macro"]["5年期LPR"]["5年期以上LPR(%)"].iloc[-1])

        PRED_STATE.update(stage="完成", pct=100,
                          result={"name": name, "hist": hist, "pred": pred,
                                  "scenarios": list(SCENARIOS.keys()),
                                  "current": cur, "forecast_end": FORECAST_END,
                                  "last": last, "drift": dr, "diag": {"rows": rows,
                                                                       "summary": summary},
                                  "macro_end": {"lpr": lpr}})
    except Exception as e:
        PRED_STATE.update(stage="失败", msg=str(e)[:300])
    finally:
        PRED_STATE["running"] = False


@app.route("/api/predict", methods=["POST"])
def api_predict():
    name = request.args.get("name", "")
    with PRED_LOCK:
        if PRED_STATE["running"]:
            return jsonify({"error": "已有预测在进行中"}), 409
        # 同步置位,避免两个并发请求同时通过检查(TOCTOU)
        PRED_STATE.update(running=True, stage="排队中…", pct=0, name=name, msg="", result=None)
    ensure_loaded()
    threading.Thread(target=_predict_job, args=(name,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/predict_progress")
def api_predict_progress():
    st = dict(PRED_STATE)
    if st["started_at"]:
        elapsed = time.time() - st["started_at"]
        st["elapsed"] = elapsed
        # ETA 外推:按已耗时间与进度估算
        pct = st["pct"]
        st["eta"] = elapsed / pct * (100 - pct) if pct > 0 else 0.0
    else:
        st["elapsed"] = 0.0
        st["eta"] = 0.0
    return jsonify(st)


# ---------------- 训练(与预测分开,带进度) ----------------
TRAIN_STATE = {"running": False, "stage": "", "done": 0, "total": 0, "msg": ""}


def _train_job():
    import time as _t
    from train_all import (build_pooled, train_parallel, save_model,
                           MODEL_FILE, TRAIN_ROUNDS, HIGH_CFG, HALF_LIFE, N_JOBS)
    from features import build_events, load_extra_events, load_macro, load_attrs
    from load_data import load_complex
    from experiment_tracker import log_experiment
    _t0 = _t.time()
    try:
        TRAIN_STATE.update(running=True, stage="解析小区数据", done=0, total=1, msg="")
        TRAIN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 "data", "train")
        paths = sorted(os.path.join(root, f) for root, _, fs in os.walk(TRAIN_DIR)
                       for f in fs if f.lower().endswith((".xlsx", ".xlsm"))
                       and not f.startswith("~$"))
        TRAIN_STATE.update(total=len(paths))
        n_workers = min(8, max(2, (os.cpu_count() or 4) // 2))
        complex_map = {}
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(load_complex, p): p for p in paths}
            for i, fut in enumerate(as_completed(futs)):
                p = futs[fut]
                try:
                    complex_map[p] = fut.result()
                except Exception:
                    pass
                if i % 10 == 0 or i == len(paths) - 1:
                    TRAIN_STATE.update(done=i + 1,
                                       stage=f"解析小区数据 {i + 1}/{len(paths)} ({n_workers} 线程)")
        complexes = [complex_map[p] for p in paths if p in complex_map]
        all_ev = pd.concat([c["events"] for c in complexes if c["events"] is not None],
                           ignore_index=True)
        events = build_events(all_ev, load_extra_events())
        macro, attrs = load_macro(), load_attrs()
        TRAIN_STATE.update(stage="构建面板特征…", done=0, total=100)
        X, y, w, meta, x_cols, attr_info = build_pooled(complexes, events, macro, attrs)
        TRAIN_STATE.update(stage=f"并行训练 {TRAIN_ROUNDS} 轮 × 5 模型", done=0, total=TRAIN_ROUNDS)
        models = train_parallel(X.values, y.values, w.values, TRAIN_ROUNDS, HIGH_CFG,
                                on_done=lambda d, t: TRAIN_STATE.update(done=d))
        save_model({"models": models, "meta": meta, "feature_cols": x_cols,
                    "has_macro": bool(macro), "attr_info": attr_info,
                    "n_rounds": TRAIN_ROUNDS, "cfg": HIGH_CFG,
                    "trained_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")})
        TRAIN_STATE.update(stage="完成", done=TRAIN_ROUNDS,
                           msg=f"训练完成: {len(meta)} 个小区,模型已保存")
        log_experiment(
            run_type="train",
            params={"n_complexes": len(complexes), "n_samples": len(X),
                    "train_rounds": TRAIN_ROUNDS, "n_estimators": HIGH_CFG.get("n_estimators"),
                    "half_life": HALF_LIFE, "n_jobs": N_JOBS, "model_file": MODEL_FILE},
            duration=_t.time() - _t0,
            notes=f"web 端训练 · 特征列数: {len(x_cols)}",
        )
    except Exception as e:
        TRAIN_STATE.update(stage="失败", msg=str(e)[:200])
    finally:
        TRAIN_STATE["running"] = False


@app.route("/api/train", methods=["POST"])
def api_train():
    with TRAIN_LOCK:
        if TRAIN_STATE["running"]:
            return jsonify({"error": "训练已在运行中"}), 409
        # 同步置位,避免并发请求(TOCTOU)
        TRAIN_STATE.update(running=True, stage="排队中…", done=0, total=0, msg="")
    threading.Thread(target=_train_job, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/train_progress")
def api_train_progress():
    return jsonify(TRAIN_STATE)


@app.route("/api/model_info")
def api_model_info():
    """模型元信息(训练时间/轮数/小区数/特征数),供前端"模型信息"卡片展示。"""
    try:
        import joblib
        from train_all import MODEL_FILE
        if not os.path.exists(MODEL_FILE):
            return jsonify({"exists": False})
        pkg = joblib.load(MODEL_FILE)
        return jsonify({
            "exists": True,
            "trained_at": pkg.get("trained_at", ""),
            "n_rounds": pkg.get("n_rounds", 0),
            "n_complexes": len(pkg.get("meta", [])),
            "n_features": len(pkg.get("feature_cols", [])),
            "has_macro": bool(pkg.get("has_macro")),
            "model_file": MODEL_FILE,
        })
    except Exception as e:
        return jsonify({"exists": False, "error": str(e)[:200]})


@app.route("/api/static_info")
def api_static_info():
    """前端硬编码文案替代:训练小区数 / 特征维数 / 训练轮数 / 树数。"""
    from train_all import TRAIN_ROUNDS, N_TREES
    n_train = len(CACHE.get("complexes", [])) if CACHE.get("loaded") else None
    n_test = len(CACHE.get("test", [])) if CACHE.get("loaded") else None
    from features import ATTRIBUTE_COLS, FEATURE_COLS
    from features import MACRO_COLS
    n_feat = len(FEATURE_COLS) + len(MACRO_COLS) + len(ATTRIBUTE_COLS) + 1  # +complex_id
    return jsonify({
        "n_train": n_train, "n_test": n_test,
        "n_features": n_feat, "train_rounds": TRAIN_ROUNDS, "n_trees": N_TREES,
        "forecast_end": FORECAST_END,
    })


if __name__ == "__main__":
    import logging
    from logging.handlers import RotatingFileHandler
    import socket

    # Web 服务日志落盘(控制台关窗后仍可追溯;5MB×3 轮转)
    try:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        os.makedirs(log_dir, exist_ok=True)
        _handler = RotatingFileHandler(
            os.path.join(log_dir, "web.log"), maxBytes=5 * 1024 * 1024,
            backupCount=3, encoding="utf-8")
        _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        app.logger.addHandler(_handler)
        app.logger.setLevel(logging.INFO)
    except Exception as e:
        print(f"[提示] Web 日志初始化失败({e}),日志仅输出控制台")

    def _open_browser(port):
        import webbrowser

        def _open():
            time.sleep(1.5)
            webbrowser.open(f"http://127.0.0.1:{port}")

        threading.Thread(target=_open, daemon=True).start()

    port = 5000
    while port < 5010:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                break
            except OSError:
                port += 1
    if port >= 5010:
        # 5000-5009 全占用时仍尝试启动(失败会打印 Flask 错误)
        port = 5000
    _open_browser(port)
    print(f"网页演示地址: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=False)
