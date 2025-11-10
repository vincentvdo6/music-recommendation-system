@echo off
echo ========================================
echo Training LightGBM Ranker with 23 Features
echo ========================================
echo.

set PYTHONPATH=%CD%

echo Step 1: Loading models and data...

REM Try to find Python
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py scripts\train_ranker.py
) else (
    where python >nul 2>&1
    if %ERRORLEVEL% EQU 0 (
        python scripts\train_ranker.py
    ) else (
        echo Error: Python not found. Please install Python or add it to PATH.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Training complete!
echo New ranker saved to: models\ranker\lightgbm_ranker.txt
echo ========================================
pause
