@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo SightBridge Spatial Awareness - Explicit environment setup
echo ============================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo The existing .venv was not found. This setup preserves it and will not create a replacement.
  exit /b 1
)

".venv\Scripts\python.exe" -c "import sys; assert sys.version_info[:3] == (3,11,9), f'Expected Python 3.11.9, found {sys.version.split()[0]}'; assert sys.prefix.lower() == __import__('os').path.abspath('.venv').lower(), f'Python is outside .venv: {sys.executable}'"
if errorlevel 1 goto fail

echo Installing or correcting CUDA-enabled PyTorch in .venv (CUDA 12.6 wheel index)...
".venv\Scripts\python.exe" -m pip install --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126"
if errorlevel 1 goto fail

echo Installing missing or incompatible project requirements into .venv...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto fail

echo Verifying Python, CUDA, and application dependencies...
".venv\Scripts\python.exe" -c "import os,sys,torch,ultralytics,transformers,cv2,pyttsx3; print('Python:',sys.executable); print('PyTorch:',torch.__version__); print('CUDA build:',torch.version.cuda); print('CUDA available:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable'); print('Ultralytics:',ultralytics.__version__); print('Transformers:',transformers.__version__); print('OpenCV:',cv2.__version__); print('pyttsx3: import OK'); assert sys.prefix.lower() == os.path.abspath('.venv').lower(), 'Python is outside .venv'; assert torch.version.cuda is not None, 'Installed PyTorch is not CUDA-enabled'; assert torch.cuda.is_available(), 'CUDA build or NVIDIA driver is unavailable'"
if errorlevel 1 goto fail

echo.
echo Setup and CUDA verification completed. No application or model weights were started.
exit /b 0

:fail
echo.
echo Setup or verification failed. The application was not started.
exit /b 1
