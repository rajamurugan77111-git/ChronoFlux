@echo off
setlocal EnableExtensions
cd /d "%~dp0"

title ChronoFlux v8
color 0B

echo ============================================================
echo   ChronoFlux v8  -  Detect ^| Forecast ^| Simulate ^| Verify
echo ============================================================
echo.
echo   Jury path after start:
echo     1. Open Simulation  -^>  Play
echo     2. Click Run live demo  (or top-bar Run demo)
echo     3. Open Forecast for T5 / T6 / T7
echo     4. Open Evidence to verify SHA-256 chain
echo.
echo   Use Google Chrome for the official demo look.
echo   URL: http://127.0.0.1:8000
echo ============================================================
echo.

if not exist "venv\Scripts\python.exe" (
    echo [1/5] Creating Python virtual environment...
    py -m venv venv
    if errorlevel 1 goto :fail
    echo [2/5] Installing requirements...
    "venv\Scripts\python.exe" -m pip install --upgrade pip
    "venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
) else (
    echo [1/5] Existing virtual environment found.
    echo [2/5] Syncing requirements...
    "venv\Scripts\python.exe" -m pip install -r requirements.txt >nul
)

echo [3/5] Ensuring LSTM world model is trained...
if not exist "models\temporal\world_model.joblib" (
    "venv\Scripts\python.exe" -m training.train_world_model
    if errorlevel 1 (
        echo World-model training warning — app will retry on startup.
    )
) else (
    echo World model weights found.
)

echo [4/5] Freeing port 8000 if a stale server is still running...
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R /C:":8000 .*LISTENING"') do (
    echo Killing PID %%P on port 8000...
    taskkill /F /PID %%P >nul 2>&1
)

echo [5/5] Starting ChronoFlux and opening Chrome...
set CHRONOFLUX_NO_BROWSER=1
start "" cmd /c "timeout /t 2 /nobreak >nul & (start chrome http://127.0.0.1:8000 || start msedge http://127.0.0.1:8000 || start http://127.0.0.1:8000)"
"venv\Scripts\python.exe" run.py
goto :end

:fail
echo.
echo Setup failed. Make sure Python 3 and the Windows py launcher are installed.
pause

:end
endlocal
