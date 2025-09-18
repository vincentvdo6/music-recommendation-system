@echo off
echo 🎵 Starting Music Recommendation System...

py --version >NUL 2>&1 || (echo ❌ Python not found! && pause && exit)

echo 📦 Installing dependencies...
py -m pip install -r requirements.txt >NUL 2>&1

echo 🚀 Server starting at http://localhost:8000
echo 🌐 Opening browser in 3 seconds...

start "" timeout /t 3 /nobreak >NUL && start http://localhost:8000

py -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload