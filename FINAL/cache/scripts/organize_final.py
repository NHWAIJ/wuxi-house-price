# -*- coding: utf-8 -*-
"""
organize_final.py — 同步 04_code 正式运行包(scripts + data + run.bat,只读副本)。
产物直接在 charts/output/comparison 查看(展示副本 01_charts 等已于 2026-08 废弃删除)。

运行: 双击 FINAL\\collect_final.bat 或 python FINAL\\scripts\\organize_final.py
"""
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # D:\BK
CACHE = os.path.join(ROOT, "FINAL", "cache")
CODE = os.path.join(CACHE, "04_code")


def sync_code():
    """刷新 04_code 正式运行包(只读副本):scripts + data + run.bat,清理旧残留。"""
    dst_s = os.path.join(CODE, "scripts")
    if os.path.isdir(dst_s):
        shutil.rmtree(dst_s)
    shutil.copytree(os.path.join(CACHE, "scripts"), dst_s,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    dst_d = os.path.join(CODE, "data")
    if os.path.isdir(dst_d):
        shutil.rmtree(dst_d)
    shutil.copytree(os.path.join(CACHE, "data"), dst_d)
    # 英文 bat(CRLF)
    bat = os.path.join(CODE, "run.bat")
    with open(bat, "w", newline="\r\n", encoding="utf-8") as f:
        f.write('@echo off\r\nchcp 65001 >nul\r\ncd /d "%~dp0"\r\n'
                'set PYTHONIOENCODING=utf-8\r\n'
                'echo ===== 04_code formal package: train + forecast =====\r\n'
                'python -u scripts\\run_joint.py > train_latest.log 2>&1\r\n'
                'type train_latest.log\r\npause >nul\r\n')
    # 清理旧残留(旧中文 bat / 旧 workbook 目录)
    for old in ("运行房价预测.bat",):
        p = os.path.join(CODE, old)
        if os.path.exists(p):
            os.remove(p)
    wb = os.path.join(CODE, "workbook")
    if os.path.isdir(wb):
        shutil.rmtree(wb)
    return "scripts + data + run.bat 已同步"


def main():
    sync_code()
    print(f"已同步正式运行包 → {CODE}")


if __name__ == "__main__":
    main()
