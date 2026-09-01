@echo off
TITLE XGBoost Fraud Analytics Setup

:: Change drive to D: and navigate to Git directory
cd /d D:\Git

:: Check if directory cloned already
IF NOT EXIST "D:\Git\Fraud-Detection" (
    echo [1/4] Cloning Fraud-Detection repository...
    git clone https://github.com/aadhitea/Fraud-Detection.git
) ELSE (
    echo Repository already exists.
)

:: Move into repository directory
cd /d D:\Git\Fraud-Detection

:: Create virtual environment if it doesn't exist
IF NOT EXIST ".venv" (
    echo [2/4] Creating Python virtual environment...
    python -m venv .venv
) ELSE (
    echo Virtual environment .venv already exists.
)

:: Activate virtual environment
echo [3/4] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install dependencies
echo [4/4] Installing requirements...
pip install --upgrade pip
if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found.
)

echo.
echo Setup Complete!
cmd /k