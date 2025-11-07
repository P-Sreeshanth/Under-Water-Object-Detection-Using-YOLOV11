#!/bin/bash

echo "==============================================="
echo "  Underwater Detection System - Setup"
echo "==============================================="
echo ""

echo "[1/3] Installing Frontend Dependencies..."
cd frontend
npm install
if [ $? -ne 0 ]; then
    echo "Failed to install frontend dependencies"
    exit 1
fi
echo "Frontend dependencies installed successfully!"
echo ""

echo "[2/3] Checking Backend..."
cd ..
if [ ! -d ".venv" ]; then
    echo "Virtual environment not found. Please run setup.sh first."
    exit 1
fi
echo "Backend is ready!"
echo ""

echo "[3/3] Setup Complete!"
echo ""
echo "==============================================="
echo "  Ready to Launch!"
echo "==============================================="
echo ""
echo "To start the application:"
echo ""
echo "1. Start Backend:"
echo "   .venv/Scripts/python.exe -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"
echo ""
echo "2. Start Frontend (in a new terminal):"
echo "   cd frontend"
echo "   npm start"
echo ""
echo "The application will be available at:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
