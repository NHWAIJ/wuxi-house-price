@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo ============================================
echo   Target Diagnose: multi-cutoff validation
echo   + formal forecast to 2031.12
echo ============================================
python -u scripts\target_diagnose.py
echo.
echo Done. Diag: output\diagnose_*.xlsx  Charts: comparison\
pause >nul
