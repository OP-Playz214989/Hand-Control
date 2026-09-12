@echo off
title Mini Jarvis
cd /d "%~dp0"

echo ==========================================================
echo           MINI JARVIS - ONE-CLICK LAUNCHER (WINDOWS)
echo ==========================================================
echo Project Directory: %~dp0
echo.

REM Check for virtual environment in .venv or venv
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_BIN=.venv\Scripts\python.exe"
    echo [OK] Using virtual environment (.venv)
) else if exist "venv\Scripts\python.exe" (
    set "PYTHON_BIN=venv\Scripts\python.exe"
    echo [OK] Using virtual environment (venv)
) else (
    set "PYTHON_BIN=python"
    echo [OK] Using system Python
)

REM Check .env file
if not exist ".env" (
    if exist ".env.example" (
        echo [INFO] .env not found. Creating from .env.example...
        copy .env.example .env >nul
        echo [OK] Created .env file.
    )
)

echo.
echo Starting Mini Jarvis...
echo ----------------------------------------------------------
echo Tips:
echo  - Show hand to webcam to control mouse cursor and gestures.
echo  - Press 'q' in the camera window or Ctrl+C to quit.
echo ----------------------------------------------------------
echo.

"%PYTHON_BIN%" main.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo ==========================================================
    echo [WARNING] Application stopped with error code %EXIT_CODE%.
    echo ==========================================================
    pause
)
