#!/bin/bash
set -e

echo "Starting Language Course Annotator..."

# Ensure venv support exists
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "python3-venv not installed. Installing..."
    sudo apt update
    sudo apt install -y python3-venv
fi

# Create venv if missing
if [ ! -d "venv" ]; then
    echo "[1/3] Creating isolated virtual environment..."
    python3 -m venv venv
fi

# Activate
source venv/bin/activate

echo "[2/3] Checking dependencies..."
pip install --upgrade pip
python -m pip install -r requirements.txt

# Ensure ffmpeg exists
if ! command -v ffmpeg &> /dev/null; then
    echo "FFmpeg not found. Installing..."
    sudo apt install -y ffmpeg
fi

echo "[3/3] Launching App..."
python main.py
