@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Accuracy Test: cutoff 2024.08, forecast 22m
echo   Progress window will pop up. Wait 3-6 min.
echo ============================================
python -u scripts\run_test.py
echo.
echo Done. Tables: output\   Charts: comparison\
pause >nul
