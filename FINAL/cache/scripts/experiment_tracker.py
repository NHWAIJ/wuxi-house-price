# -*- coding: utf-8 -*-
"""
experiment_tracker.py — 实验追踪

每次训练/预测运行自动记录关键参数和结果到 experiments.csv,
便于回溯每次运行的信息(参数、数据量、分数、耗时等)。

使用方法:
    from experiment_tracker import log_experiment
    log_experiment(run_type="train", params={...}, metrics={...}, duration=123)
"""
import os
import csv
from datetime import datetime


EXPERIMENTS_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments.csv",
)

_FIELDS = [
    "run_id",          # 自增编号
    "timestamp",       # 运行时间
    "run_type",        # train / predict / diagnose / test
    "duration_sec",    # 耗时(秒)
    "n_complexes",     # 小区数
    "n_samples",       # 样本行数
    "train_rounds",    # 训练轮数
    "n_estimators",    # 树数
    "half_life",       # 衰减半衰期
    "n_jobs",          # 并行进程数
    "model_file",      # 模型文件路径
    "score_bk",        # 北控雁栖湖得分
    "score_dby",       # 新力帝泊湾得分
    "notes",           # 备注
]


def _next_run_id():
    """自增 run_id。"""
    if not os.path.exists(EXPERIMENTS_FILE):
        return 1
    try:
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            max_id = 0
            for row in reader:
                try:
                    max_id = max(max_id, int(row.get("run_id", 0)))
                except (ValueError, TypeError):
                    pass
            return max_id + 1
    except Exception:
        return 1


def log_experiment(run_type="", params=None, metrics=None, duration=0, notes=""):
    """
    记录一次实验。

    参数:
        run_type: "train" / "predict" / "diagnose" / "test"
        params: dict, 训练/预测参数
        metrics: dict, 结果指标(如 score_bk, score_dby)
        duration: 耗时(秒)
        notes: 备注文本
    """
    if params is None:
        params = {}
    if metrics is None:
        metrics = {}
    row = {
        "run_id": _next_run_id(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_type": run_type,
        "duration_sec": f"{duration:.0f}",
        "n_complexes": params.get("n_complexes", ""),
        "n_samples": params.get("n_samples", ""),
        "train_rounds": params.get("train_rounds", ""),
        "n_estimators": params.get("n_estimators", ""),
        "half_life": params.get("half_life", ""),
        "n_jobs": params.get("n_jobs", ""),
        "model_file": params.get("model_file", ""),
        "score_bk": metrics.get("score_bk", ""),
        "score_dby": metrics.get("score_dby", ""),
        "notes": notes,
    }
    _append_row(row)


def _append_row(row):
    """追加一行到 CSV。"""
    exists = os.path.exists(EXPERIMENTS_FILE)
    try:
        os.makedirs(os.path.dirname(EXPERIMENTS_FILE), exist_ok=True)
        with open(EXPERIMENTS_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        print(f"[experiment_tracker] 写入失败: {e}")


def get_summary():
    """返回最近 10 条实验记录列表。"""
    if not os.path.exists(EXPERIMENTS_FILE):
        return []
    try:
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        return rows[-10:]
    except Exception:
        return []