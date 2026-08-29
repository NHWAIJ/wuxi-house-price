@echo off
chcp 65001 >nul
cd /d "%~dp0cache"
echo ============================================
echo   Building: 正式安装包 setup.exe (Inno Setup)
echo ============================================
set ISCC="%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% set ISCC="C:\Program Files\Inno Setup 6\ISCC.exe"
%ISCC% installer_script.iss
if errorlevel 1 goto :err
echo.
echo Done! Output: dist_setup\房价预测系统_安装包.exe
pause >nul
exit /b 0
:err
echo.
echo Build failed. See messages above.
pause >nul
exit /b 1
