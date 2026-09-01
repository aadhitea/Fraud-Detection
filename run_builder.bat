@echo off
TITLE Fraud Detection Project Auto-Builder

:: Step 1: Navigate to target directory
cd /d D:\Git
echo [1/5] Navigating to D:\Git...

:: Step 2: Run Python script to build Fraud-Detection.zip
echo [2/5] Running Python builder script...
python generate_project.py

IF NOT EXIST "Fraud-Detection.zip" (
    echo Error: Fraud-Detection.zip was not generated. Check Python installation.
    pause
    exit /b
)

:: Step 3: Extract the ZIP archive
echo [3/5] Extracting Fraud-Detection.zip...
powershell -Command "Expand-Archive -Path 'Fraud-Detection.zip' -DestinationPath 'D:\Git' -Force"

:: Step 4: Navigate into project folder & create virtual environment
cd /d D:\Git\Fraud-Detection

echo [4/5] Creating Python virtual environment (.venv)...
IF NOT EXIST ".venv" (
    python -m venv .venv
) ELSE (
    echo Virtual environment already exists.
)

:: Step 5: Activate environment & install requirements
echo [5/5] Activating environment and installing dependencies...
call .venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ===================================================
echo Project setup complete!
echo Active Directory: D:\Git\Fraud-Detection
echo Virtual Environment: Active (.venv)
echo ===================================================
echo.

cmd /k