@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
set PYTHONIOENCODING=utf-8
echo Starting web demo ... browser opens automatically.
echo Close this window to stop the server.
python -u web\app.py
pause >nul
