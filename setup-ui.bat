@echo off
echo ===============================================
echo   Underwater Detection System - Setup
echo ===============================================
echo.

echo [1/3] Installing Frontend Dependencies...
cd frontend
call npm install
if errorlevel 1 (
    echo Failed to install frontend dependencies
    pause
    exit /b 1
)
echo Frontend dependencies installed successfully!
echo.

echo [2/3] Checking Backend...
cd ..
if not exist ".venv" (
    echo Virtual environment not found. Please run setup.sh first.
    pause
    exit /b 1
)
echo Backend is ready!
echo.

echo [3/3] Setup Complete!
echo.
echo ===============================================
echo   Ready to Launch!
echo ===============================================
echo.
echo To start the application:
echo.
echo 1. Start Backend:
echo    .venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
echo.
echo 2. Start Frontend (in a new terminal):
echo    cd frontend
echo    npm start
echo.
echo The application will be available at:
echo    Frontend: http://localhost:3000
echo    Backend:  http://localhost:8000
echo.
pause
