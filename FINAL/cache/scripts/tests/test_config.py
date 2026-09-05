# -*- coding: utf-8 -*-
"""
config_loader 测试

运行: python -m pytest scripts/tests/test_config.py -v
"""
import os
import sys
import tempfile
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import _DEFAULTS, _merge, load_config


def test_defaults_are_complete():
    """默认配置应包含所有顶级键。"""
    for key in ["train", "models", "predict", "diagnose", "drift", "test"]:
        assert key in _DEFAULTS, f"缺少顶级配置键: {key}"


def test_train_config_has_required_keys():
    """训练配置应包含所有必需键。"""
    tc = _DEFAULTS["train"]
    for key in ["rounds", "n_estimators", "half_life", "min_history",
                 "cutoff", "high_cfg", "n_jobs"]:
        assert key in tc, f"训练配置缺少键: {key}"
    # high_cfg 子键
    for key in ["rf_depth", "gbr_depth", "lr", "min_leaf"]:
        assert key in tc["high_cfg"], f"high_cfg 缺少键: {key}"


def test_predict_config():
    """预测配置应包含所有必需键。"""
    pc = _DEFAULTS["predict"]
    for key in ["forecast_end", "forecast_months", "interval_quantiles",
                 "interval_scale", "n_rounds", "scenarios"]:
        assert key in pc, f"预测配置缺少键: {key}"
    # 区间参数
    assert len(pc["interval_quantiles"]) == 2
    assert pc["interval_quantiles"][0] < pc["interval_quantiles"][1]
    # 情景
    for key in ["baseline", "optimistic", "pessimistic"]:
        assert key in pc["scenarios"], f"情景缺少键: {key}"


def test_merge_overrides():
    """_merge 应正确合并嵌套字典。"""
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 4}
    result = _merge(base, override)
    assert result["a"] == 1
    assert result["b"]["c"] == 99      # 被覆盖
    assert result["b"]["d"] == 3        # 保留
    assert result["e"] == 4             # 新增


def test_load_config_yaml():
    """load_config 应读取 yaml 并合并到默认值。"""
    yaml_content = """
train:
  rounds: 50
  n_estimators: 300
predict:
  forecast_end: "2030-12"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml",
                                     delete=False, encoding="utf-8") as f:
        f.write(yaml_content)
        tmp_path = f.name

    try:
        # 模拟环境变量指向临时文件
        old_env = os.environ.get("BK_CONFIG")
        os.environ["BK_CONFIG"] = tmp_path
        cfg = load_config()
        assert cfg["train"]["rounds"] == 50
        assert cfg["train"]["n_estimators"] == 300
        assert cfg["train"]["half_life"] == 24  # 来自默认值
        assert cfg["predict"]["forecast_end"] == "2030-12"
        if old_env is None:
            del os.environ["BK_CONFIG"]
        else:
            os.environ["BK_CONFIG"] = old_env
    finally:
        os.unlink(tmp_path)


def test_diagnose_config():
    """诊断配置应包含所有必需键。"""
    dc = _DEFAULTS["diagnose"]
    assert dc["n_cutoffs"] >= 2
    assert dc["n_window"] >= 6


def test_drift_config():
    """漂移检测配置应包含所有必需键。"""
    drc = _DEFAULTS["drift"]
    assert drc["high_threshold"] < drc["medium_threshold"]
    assert drc["min_samples"] >= 6
    assert drc["min_dist_samples"] >= 5