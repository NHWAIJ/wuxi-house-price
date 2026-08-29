@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo ===== 04_code formal package: train + forecast =====
python -u scripts\run_joint.py > train_latest.log 2>&1
type train_latest.log
pause >nul
