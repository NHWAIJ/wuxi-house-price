@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Joint Training: data/train all complexes
echo   ONE progress window. Wait 10-12 min.
echo ============================================
python -u scripts\run_joint.py > logs\train_latest.log 2>&1
type logs\train_latest.log
echo.
echo Done. Models: models\  Tables: output\  Charts: comparison\
pause >nul
