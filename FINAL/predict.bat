@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Forecast only (Holt main for targets)
echo   Progress window will pop up. Wait 1-2 min.
echo ============================================
python -u scripts\predict.py
echo.
echo Done. Tables: output\  Charts: comparison\
pause >nul
