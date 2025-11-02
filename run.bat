@echo off
setlocal enabledelayedexpansion

:: Check Python version
python --version > nul 2>&1
if errorlevel 1 (
    echo Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Check if the virtual environment folder exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

:: Activate the virtual environment
call venv\Scripts\activate

:: Upgrade pip
python -m pip install --upgrade pip

:: Install PyTorch with CUDA (adjust CUDA version as needed)
echo Installing PyTorch with CUDA...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

:: Install other requirements
echo Installing additional requirements...
pip install -r requirements.txt

:: Verify installations
echo Verifying PyTorch CUDA installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda)"

:: Run the Python program
echo Running vision.py...
python vision.py

:: Deactivate the virtual environment
deactivate

:: Keep console window open
pause