#!/bin/bash

# Setup script for Underwater Image Analysis API
# This script sets up the development environment

echo "=================================="
echo "Underwater Image Analysis API"
echo "Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
if [ $? -eq 0 ]; then
    echo "✓ Python found: $python_version"
else
    echo "✗ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "⚠ Virtual environment already exists. Skipping..."
else
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✓ Requirements installed successfully"
else
    echo "✗ Failed to install requirements"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p models
mkdir -p static
mkdir -p logs
echo "✓ Directories created"

# Copy environment file
echo ""
if [ ! -f ".env" ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created. Please edit it with your configuration."
else
    echo "⚠ .env file already exists. Skipping..."
fi

# Check for model files
echo ""
echo "Checking for model files..."
if [ ! -f "models/best.pt" ]; then
    echo "⚠ YOLOv11 model not found at models/best.pt"
    echo "  You need to download or train a YOLOv11 model."
    echo "  Run: python download_models.py"
else
    echo "✓ YOLOv11 model found"
fi

if [ ! -f "models/enhancer_model.pth" ]; then
    echo "⚠ Enhancement model not found at models/enhancer_model.pth"
    echo "  The API will work without enhancement (detection only)."
else
    echo "✓ Enhancement model found"
fi

# Test installation
echo ""
echo "Testing installation..."
python -c "import fastapi, torch, ultralytics, cv2, numpy; print('✓ All core packages imported successfully')"
if [ $? -eq 0 ]; then
    echo "✓ Installation test passed"
else
    echo "✗ Installation test failed"
    exit 1
fi

# Summary
echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Download models (if needed): python download_models.py"
echo "  3. Edit .env file with your configuration"
echo "  4. Start the server: python app/main.py"
echo "  5. Visit: http://localhost:8000/docs"
echo ""
echo "For testing:"
echo "  - Run tests: pytest test_api.py -v"
echo "  - Try example: python example_usage.py"
echo ""
