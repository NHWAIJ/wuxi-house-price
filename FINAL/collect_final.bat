@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Sync 04_code formal package (scripts+data)
echo ============================================
python -u scripts\organize_final.py
echo.
pause >nul
