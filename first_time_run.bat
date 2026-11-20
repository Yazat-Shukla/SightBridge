@echo off
setlocal EnableExtensions

rem The outer invocation runs this batch through PowerShell Tee-Object so all
rem child stdout/stderr stays visible and is also persisted to one local log.
if /i "%~1"=="__logged" goto fresh_setup
set "SIGHTBRIDGE_SETUP_BODY_STARTED="
goto start_logging

:fresh_setup
if defined SIGHTBRIDGE_SETUP_BODY_STARTED goto duplicate_setup_invocation
set "SIGHTBRIDGE_SETUP_BODY_STARTED=1"
for %%I in ("%~dp0.") do set "PROJECT_DIR=%%~fI"
cd /d "%PROJECT_DIR%"
set "YOLO_AUTOINSTALL=False"
set "HF_HOME=%PROJECT_DIR%\models\huggingface"
set "HF_HUB_OFFLINE=0"
set "TRANSFORMERS_OFFLINE=0"
set "DEPTH_MODEL_ID=depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "FAILED_STAGE=Preflight"
set "FAILED_COMMAND=Validate required project files"
set "FAIL_CODE=1"
set "SIGHTBRIDGE_FRESH_VENV_CREATED="

echo [%DATE% %TIME%] [SCRIPT: first_time_run.bat fresh-setup flow v2]
echo [%DATE% %TIME%] [SCRIPT PATH: %~f0]
call :stage "Preflight" "Check required project files"
if not exist "%PROJECT_DIR%\app.py" goto app_missing
if not exist "%PROJECT_DIR%\requirements.txt" goto requirements_missing
if not exist "%PROJECT_DIR%\yolo11n.pt" goto yolo_missing
if not exist "%PROJECT_DIR%\bytetrack-sightbridge.yaml" goto tracker_missing

set "FAILED_STAGE=Python prerequisite"
set "FAILED_COMMAND=where py"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
where py
if errorlevel 1 (set "FAILED_COMMAND=where py (Windows Python launcher check)"& set "FAIL_CODE=%ERRORLEVEL%"& goto python_missing)
set "FAILED_COMMAND=py -3.11 -c Python 3.11 x64 verification"
py -3.11 -c "import sys,struct; assert sys.version_info[:2]==(3,11), 'Python 3.11 required'; assert struct.calcsize('P')==8, '64-bit Python required'"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto python_missing)

if /i not "%VENV_DIR%"=="%PROJECT_DIR%\.venv" goto unsafe_venv
call :stage "Environment deletion" "Delete only %VENV_DIR% if it exists"
if exist "%VENV_DIR%" (
  set "FAILED_STAGE=Environment deletion"
  set "FAILED_COMMAND=rmdir /s /q %VENV_DIR%"
  rmdir /s /q "%VENV_DIR%"
  if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto venv_delete_failed)
  if exist "%VENV_DIR%" goto venv_delete_failed
  echo [%DATE% %TIME%] [RESULT: Project-local .venv deleted]
) else (
  echo [%DATE% %TIME%] [RESULT: No existing project-local .venv to delete]
)

set "FAILED_STAGE=Create virtual environment"
set "FAILED_COMMAND=py -3.11 -m venv %VENV_DIR%"
call :stage "Create virtual environment" "py -3.11 -m venv %VENV_DIR%"
py -3.11 -m venv "%VENV_DIR%"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto venv_create_failed)
if not exist "%VENV_DIR%\Scripts\python.exe" goto venv_create_failed
set "SIGHTBRIDGE_FRESH_VENV_CREATED=1"
echo [%DATE% %TIME%] [RESULT: New project-local Python 3.11 virtual environment created]

if not "%SIGHTBRIDGE_FRESH_VENV_CREATED%"=="1" goto fresh_environment_guard_failed

set "FAILED_STAGE=CUDA PyTorch installation"
set "FAILED_COMMAND=%VENV_DIR%\Scripts\python.exe -B -m pip install --disable-pip-version-check --timeout 30 --retries 3 --index-url https://download.pytorch.org/whl/cu126 torch==2.14.1+cu126 torchvision==0.29.1+cu126"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -m pip install --disable-pip-version-check --timeout 30 --retries 3 --index-url https://download.pytorch.org/whl/cu126 "torch==2.14.1+cu126" "torchvision==0.29.1+cu126"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto failed)

set "FAILED_STAGE=Project dependency installation"
set "FAILED_COMMAND=%VENV_DIR%\Scripts\python.exe -B -m pip install --disable-pip-version-check --timeout 30 --retries 3 -r %PROJECT_DIR%\requirements.txt"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -m pip install --disable-pip-version-check --timeout 30 --retries 3 -r "%PROJECT_DIR%\requirements.txt"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto failed)

set "FAILED_STAGE=Python and CUDA verification"
set FAILED_COMMAND="%VENV_DIR%\Scripts\python.exe" -B -c "import sys,struct,torch,cv2,ultralytics,transformers,PIL,pyttsx3; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8; assert torch.version.cuda is not None and torch.cuda.is_available()"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -c "import sys,struct,torch,cv2,ultralytics,transformers,PIL,pyttsx3; assert sys.version_info[:2]==(3,11) and struct.calcsize('P')==8, 'Python 3.11 x64 required'; assert torch.version.cuda is not None and torch.cuda.is_available(), 'CUDA 12.6 or compatible NVIDIA driver unavailable'; print('Python:',sys.version.split()[0]); print('CUDA:',torch.version.cuda); print('GPU:',torch.cuda.get_device_name(0))"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto cuda_failed)

call :stage "Project-local model directories" "Create %PROJECT_DIR%\models and %HF_HOME% if missing"
if not exist "%PROJECT_DIR%\models" mkdir "%PROJECT_DIR%\models"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto model_dir_failed)
if not exist "%PROJECT_DIR%\models\huggingface" mkdir "%PROJECT_DIR%\models\huggingface"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto model_dir_failed)

set "FAILED_STAGE=Indoor depth checkpoint download"
set FAILED_COMMAND="%VENV_DIR%\Scripts\python.exe" -B -c "import os; from huggingface_hub import snapshot_download; snapshot_download('%DEPTH_MODEL_ID%', cache_dir=os.environ['HF_HOME'])"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -c "import os; from huggingface_hub import snapshot_download; p=snapshot_download('%DEPTH_MODEL_ID%', cache_dir=os.environ['HF_HOME']); print('Checkpoint snapshot:',p)"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto depth_download_failed)

set "FAILED_STAGE=Local depth checkpoint verification"
set "FAILED_COMMAND=%VENV_DIR%\Scripts\python.exe -B -c load indoor checkpoint with local_files_only=True"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -c "import os; from huggingface_hub import snapshot_download; from transformers import AutoImageProcessor,AutoModelForDepthEstimation; p=snapshot_download('%DEPTH_MODEL_ID%', cache_dir=os.environ['HF_HOME'], local_files_only=True); processor=AutoImageProcessor.from_pretrained(p, local_files_only=True); model=AutoModelForDepthEstimation.from_pretrained(p, local_files_only=True); print('Local checkpoint and processor load verified:',p); del model,processor"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto depth_verify_failed)

if not exist "%PROJECT_DIR%\yolo11n.pt" goto yolo_missing
set "FAILED_STAGE=Local YOLO weights verification"
set "FAILED_COMMAND=%VENV_DIR%\Scripts\python.exe -B -c load local yolo11n.pt"
call :stage "%FAILED_STAGE%" "%FAILED_COMMAND%"
"%VENV_DIR%\Scripts\python.exe" -B -c "from ultralytics import YOLO; m=YOLO('yolo11n.pt'); print('YOLO model loaded locally: yolo11n.pt'); del m"
if errorlevel 1 (set "FAIL_CODE=%ERRORLEVEL%"& goto yolo_verify_failed)

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"
if not "%SIGHTBRIDGE_FRESH_VENV_CREATED%"=="1" goto fresh_environment_guard_failed
call :stage "Application launch" "%VENV_DIR%\Scripts\python.exe %PROJECT_DIR%\app.py --source 0 --depth-environment indoor"
echo First-time setup and verification completed. Launching SightBridge...
"%VENV_DIR%\Scripts\python.exe" "%PROJECT_DIR%\app.py" --source 0 --depth-environment indoor
set "APP_EXIT=%ERRORLEVEL%"
if not "%APP_EXIT%"=="0" echo SightBridge exited with code %APP_EXIT%.
pause
exit /b %APP_EXIT%

:stage
set "FAILED_STAGE=%~1"
set "FAILED_COMMAND=%~2"
echo.
echo [%DATE% %TIME%] [STAGE: %~1]
echo [%DATE% %TIME%] [COMMAND: %~2]
exit /b 0

:app_missing
set "FAILED_COMMAND=Test for %PROJECT_DIR%\app.py"
goto missing_file
:requirements_missing
set "FAILED_COMMAND=Test for %PROJECT_DIR%\requirements.txt"
goto missing_file
:yolo_missing
set "FAILED_COMMAND=Test for %PROJECT_DIR%\yolo11n.pt"
goto missing_file
:tracker_missing
set "FAILED_COMMAND=Test for %PROJECT_DIR%\bytetrack-sightbridge.yaml"
goto missing_file
:missing_file
set "FAIL_CODE=1"
goto failed
:python_missing
set "FAILED_STAGE=Python prerequisite"
goto failed
:unsafe_venv
set "FAILED_STAGE=Virtual environment safety check"
set "FAILED_COMMAND=Validate VENV_DIR is exactly the project-local .venv"
set "FAIL_CODE=1"
goto failed
:venv_delete_failed
set "FAILED_STAGE=Environment recreation"
set "FAILED_COMMAND=rmdir /s /q %VENV_DIR%"
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:venv_create_failed
set "FAILED_STAGE=Create virtual environment"
set "FAILED_COMMAND=py -3.11 -m venv %VENV_DIR%"
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:fresh_environment_guard_failed
set "FAILED_STAGE=Fresh environment guard"
set "FAILED_COMMAND=Require this invocation to create %VENV_DIR% before installation or launch"
set "FAIL_CODE=1"
goto failed
:duplicate_setup_invocation
echo ERROR: This setup process has already entered the fresh-setup path once.
echo Refusing a second pass to prevent the environment from being recreated again.
exit /b 1
:cuda_failed
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:model_dir_failed
set "FAILED_STAGE=Project-local model directories"
set "FAILED_COMMAND=mkdir %PROJECT_DIR%\models\huggingface"
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:depth_download_failed
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:depth_verify_failed
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:yolo_verify_failed
if not defined FAIL_CODE set "FAIL_CODE=1"
goto failed
:failed
echo.
echo [%DATE% %TIME%] [FAILURE]
echo Failed setup stage: %FAILED_STAGE%
echo Failed command: %FAILED_COMMAND%
echo Exit code: %FAIL_CODE%
echo Detailed log: %PROJECT_DIR%\logs\first_time_setup.log
echo SightBridge was not launched.
pause
exit /b %FAIL_CODE%

:start_logging
set "PROJECT_DIR=%~dp0."
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
set "LOG_DIR=%PROJECT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\first_time_setup.log"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if errorlevel 1 (
  echo ERROR: Cannot create project-local log directory: %LOG_DIR%
  pause
  exit /b 1
)
set "SIGHTBRIDGE_SETUP_SCRIPT=%~f0"
set "SIGHTBRIDGE_SETUP_LOG=%LOG_FILE%"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$log=$env:SIGHTBRIDGE_SETUP_LOG; $script=$env:SIGHTBRIDGE_SETUP_SCRIPT; [System.IO.File]::WriteAllText($log, ('[' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff') + '] [SETUP SESSION START]' + [Environment]::NewLine)); & $env:ComSpec /d /c ('call "' + $script + '" __logged & exit') 2>&1 | ForEach-Object { $line=[string]$_; Write-Output $line; [System.IO.File]::AppendAllText($log, $line + [Environment]::NewLine) }; exit $LASTEXITCODE"
set "WRAPPER_EXIT=%ERRORLEVEL%"
if not "%WRAPPER_EXIT%"=="0" (
  echo Setup failed. Review detailed log: %LOG_FILE%
  pause
)
exit /b %WRAPPER_EXIT%
