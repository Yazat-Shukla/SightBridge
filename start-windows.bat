@echo off
setlocal
cd /d "%~dp0"
set "YOLO_AUTOINSTALL=False"
echo ============================================================
echo SightBridge Spatial Awareness - Launch
echo ============================================================
echo.
if not exist ".venv\Scripts\python.exe" (
  echo Project environment not found. Run setup-windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -c "import sys; assert sys.prefix.lower() == __import__('os').path.abspath('.venv').lower(), 'Wrong Python environment'" >nul 2>nul
if errorlevel 1 (
  echo Project environment is invalid. Run setup-windows.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 (
  echo.
  echo Application exited with an error. Read README.md for troubleshooting.
  pause
  exit /b 1
)
pause
exit /b 0
