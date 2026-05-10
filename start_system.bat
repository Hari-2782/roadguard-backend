@echo off
echo ================================
echo AI ROAD SAFETY PLATFORM - LAUNCHER
echo ================================

REM Move to project directory
cd /d "%~dp0"

echo [1/2] Starting API Server (Backend control)...
start "AI API Server" cmd /k "..\.venv\Scripts\python backend/api/api_server.py"

echo [2/2] Waiting for API...
timeout /t 3 >nul

echo [DONE] Opening Dashboard...
start http://localhost:5000/frontend/user/dashboard.html

echo.
echo ==================================================
echo  SYSTEM READY!
echo  Go to the Dashboard in your browser.
echo  Click "START LIVE CAMERA" to begin detection.
echo ==================================================
pause