#!/bin/bash

# Quick Start Training Script for Aquarium Dataset
# This script will prepare everything and start training

echo "======================================================================"
echo "UNDERWATER IMAGE ANALYSIS - QUICK START TRAINING"
echo "======================================================================"
echo ""

# Change to project directory
cd "/home/campus/Downloads/majot team1/underwater_analysis"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
python3 -c "import ultralytics" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies (this may take 5-10 minutes)..."
    pip install --quiet torch torchvision ultralytics fastapi uvicorn \
        python-multipart opencv-python numpy pillow pydantic pydantic-settings \
        tqdm matplotlib pyyaml
    echo "✓ Dependencies installed"
else
    echo "✓ Dependencies already installed"
fi

# Check CUDA availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Start training
echo ""
echo "======================================================================"
echo "STARTING TRAINING"
echo "======================================================================"
echo ""
python3 train_aquarium.py

echo ""
echo "======================================================================"
echo "Training script completed!"
echo "======================================================================"
