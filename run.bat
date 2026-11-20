@echo off
setlocal
for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
cd /d "%PROJECT_DIR%"
set "YOLO_AUTOINSTALL=False"
set "HF_HOME=%PROJECT_DIR%\models\huggingface"
set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

echo ============================================================
echo SightBridge Spatial Awareness - Launch
echo ============================================================
echo.

if not exist "%PROJECT_DIR%\.venv\Scripts\python.exe" goto setup_missing
if not exist "%PROJECT_DIR%\app.py" goto app_missing
if not exist "%PROJECT_DIR%\yolo11n.pt" goto yolo_missing

"%PROJECT_DIR%\.venv\Scripts\python.exe" -B -c "import sys,struct,torch,cv2,ultralytics,transformers,PIL,pyttsx3; assert sys.version_info[:2]==(3,11), 'Python 3.11 required'; assert struct.calcsize('P')==8, '64-bit Python required'; assert torch.version.cuda is not None and torch.cuda.is_available(), 'CUDA-enabled PyTorch and compatible NVIDIA driver required'"
if errorlevel 1 goto environment_invalid

"%PROJECT_DIR%\.venv\Scripts\python.exe" -B -c "import os; from huggingface_hub import snapshot_download; snapshot_download('depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf', cache_dir=os.environ['HF_HOME'], local_files_only=True)"
if errorlevel 1 goto depth_missing

echo Launching SightBridge with the project-local environment and model cache...
if /i "%~1"=="debug" (
  "%PROJECT_DIR%\.venv\Scripts\python.exe" "%PROJECT_DIR%\app.py" --source 0 --depth-environment indoor --vision-debug --depth-diagnostics --speech-debug
) else (
  "%PROJECT_DIR%\.venv\Scripts\python.exe" "%PROJECT_DIR%\app.py" --source 0 --depth-environment indoor
)
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" echo SightBridge exited with code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:setup_missing
echo ERROR: Project .venv is missing. Please run first_time_run.bat first.
goto failed
:app_missing
echo ERROR: app.py was not found in the project folder.
goto failed
:yolo_missing
echo ERROR: yolo11n.pt is missing. Copy the provided local model into the project root.
goto failed
:environment_invalid
echo ERROR: Project dependencies, Python 3.11 x64, CUDA PyTorch, or NVIDIA driver verification failed.
echo Please run first_time_run.bat first and check that this PC has a compatible NVIDIA driver.
goto failed
:depth_missing
echo ERROR: The project-local indoor Depth Anything checkpoint is missing or incomplete.
echo Please run first_time_run.bat with working internet access.
goto failed
:failed
pause
exit /b 1
