@echo off
echo Starting Language Course Annotator...
echo.

:: Check if python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH! Please install Python 3.10+.
    pause
    exit /b
)

:: Create virtual environment if missing
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Creating isolated virtual environment...
    python -m venv venv
)

:: Activate the environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo [2/4] Updating pip...
python -m pip install --upgrade pip >nul

:: Install dependencies
echo [3/4] Checking dependencies...
python -m pip install -r requirements.txt

:: Ensure FFmpeg exists
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: FFmpeg not found.
    echo Please install FFmpeg and ensure it is in your PATH.
    echo Download from: https://ffmpeg.org/download.html
    echo.
)

:: Launch App
echo [4/4] Launching App...
python main.py

pause
