# -*- coding: utf-8 -*-
"""
config_loader.py — 集中配置加载器

从 FINAL/cache/config.yaml 加载全局配置,缺失参数自动回退到代码内默认值。
所有脚本统一通过此模块读取配置,避免散落各处的硬编码常量。

用法:
    from config_loader import CFG
    rounds = CFG["train"]["rounds"]
    quantiles = CFG["predict"]["interval_quantiles"]
"""
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None


# ============================================================
# 默认配置(与代码内现有硬编码值完全一致)
# ============================================================
_DEFAULTS = {
    "train": {
        "rounds": 100,
        "n_estimators": 500,
        "half_life": 24,
        "min_history": 12,
        "cutoff": "2024-08-31",
        "high_cfg": {
            "rf_depth": 5,
            "gbr_depth": 4,
            "lr": 0.02,
            "min_leaf": 3,
        },
        "n_jobs": "auto",
    },
    "models": {
        "horizon": 12,
        "n_rounds": 30,
        "walk_step": 6,
        "default": {
            "n_estimators": 400,
            "rf_depth": 4,
            "gbr_depth": 3,
            "lr": 0.03,
            "min_leaf": 4,
        },
    },
    "predict": {
        "forecast_end": "2031-12",
        "forecast_months": 66,
        "interval_quantiles": [25, 75],
        "interval_scale": 0.5,
        "n_rounds": 30,
        "scenarios": {
            "baseline": 36,
            "optimistic": 18,
            "pessimistic": 72,
        },
    },
    "diagnose": {
        "n_cutoffs": 4,
        "n_window": 12,
    },
    "drift": {
        "high_threshold": 5,
        "medium_threshold": 15,
        "min_samples": 13,
        "min_dist_samples": 10,
    },
    "test": {
        "cutoff": "2024-08-31",
        "n_months": 22,
        "n_rounds": 30,
        "use_holt_main": True,
        "use_calibration": True,
    },
}


def _find_config():
    """从脚本目录向上查找 config.yaml。"""
    # 搜索路径优先级:环境变量显式指定 > 脚本目录父目录(cache/) > 当前工作目录
    candidates = [
        os.environ.get("BK_CONFIG", ""),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "config.yaml"),
        os.path.join(os.getcwd(), "config.yaml"),
    ]
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _merge(base, override):
    """递归合并字典,override 覆盖 base。"""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config():
    """加载配置,返回 dict。"""
    cfg_path = _find_config()
    if cfg_path is None or yaml is None:
        return _DEFAULTS
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        return _merge(_DEFAULTS, user_cfg)
    except Exception as e:
        print(f"[config] 加载 config.yaml 失败({e}),使用默认配置")
        return _DEFAULTS


# 全局单例(模块导入时即加载)
CFG = load_config()