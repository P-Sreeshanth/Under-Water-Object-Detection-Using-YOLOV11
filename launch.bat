@echo off
title Underwater Detection System - Launcher
color 0B

echo.
echo  ====================================================
echo   UNDERWATER OBJECT DETECTION SYSTEM
echo   Dual-Model AI Detection with Stunning UI
echo  ====================================================
echo.
echo  [1] Launch Backend Only
echo  [2] Launch Frontend Only  
echo  [3] Launch Full System (Backend + Frontend)
echo  [4] Setup/Install Dependencies
echo  [5] Exit
echo.
set /p choice="Select option (1-5): "

if "%choice%"=="1" goto backend
if "%choice%"=="2" goto frontend
if "%choice%"=="3" goto fullsystem
if "%choice%"=="4" goto setup
if "%choice%"=="5" exit
goto invalid

:backend
echo.
echo [BACKEND] Starting FastAPI server...
echo Backend will be available at: http://localhost:8000
echo.
.venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
pause
exit

:frontend
echo.
echo [FRONTEND] Starting React development server...
echo Frontend will be available at: http://localhost:3000
echo.
cd frontend
start cmd /k npm start
cd ..
pause
exit

:fullsystem
echo.
echo [FULL SYSTEM] Launching Backend and Frontend...
echo.
echo Starting Backend Server...
start "Backend - Port 8000" cmd /k ".venv\Scripts\python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
cd frontend
start "Frontend - Port 3000" cmd /k "npm start"
cd ..

echo.
echo ====================================================
echo  System Launched Successfully!
echo ====================================================
echo.
echo  Backend:  http://localhost:8000
echo  Frontend: http://localhost:3000
echo  Docs:     http://localhost:8000/docs
echo.
echo  Both servers are running in separate windows.
echo  Close this window when done.
echo ====================================================
echo.
pause
exit

:setup
echo.
echo [SETUP] Installing dependencies...
echo.
call setup-ui.bat
goto menu

:invalid
echo.
echo Invalid option. Please try again.
timeout /t 2 /nobreak >nul
cls
goto menu

:menu
cls
goto :eof
