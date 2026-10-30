@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   ChronoFlux - Download real UNSW-NB15 + retrain models
echo ============================================================

if not exist "venv\Scripts\python.exe" (
    echo Creating venv...
    py -m venv venv
    if errorlevel 1 goto :fail
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
)

echo [1/2] Downloading and preparing published UNSW-NB15 split...
"venv\Scripts\python.exe" data_tools\prepare_real_unsw_dataset.py
if errorlevel 1 goto :fail

echo [2/2] Retraining detection, stage and anomaly models...
"venv\Scripts\python.exe" -m training.train_all
if errorlevel 1 goto :fail

echo.
echo DONE. Real benchmark data and models are ready.
pause
goto :end

:fail
echo.
echo Operation failed. Check your internet connection and terminal output.
pause

:end
endlocal
