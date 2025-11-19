@echo off
echo 🎵 Starting Music Recommendation System...
echo.

REM Find Python installation
set PYTHON_EXE=
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_EXE=py
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe
    goto :found_python
)

if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe
    goto :found_python
)

echo ❌ Python not found!
echo Please install Python or add it to PATH.
echo.
pause
exit /b 1

:found_python
echo ✅ Using Python: %PYTHON_EXE%
echo.

REM Uncomment the lines below if you need to install/update dependencies
REM echo 📦 Installing dependencies...
REM %PYTHON_EXE% -m pip install -r requirements.txt
REM if %ERRORLEVEL% NEQ 0 (
REM     echo ❌ Failed to install dependencies!
REM     pause
REM     exit /b 1
REM )
REM echo ✅ Dependencies installed
REM echo.

echo 🚀 Server starting at http://localhost:8000
echo 📝 Press CTRL+C to stop the server
echo 🌐 Opening browser in 3 seconds...
echo.

start "" cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

%PYTHON_EXE% -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

REM If server exits, pause to see error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Server exited with error code %ERRORLEVEL%
    pause
)
