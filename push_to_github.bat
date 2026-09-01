@echo off
TITLE GitHub Push via Browser Login

:: Navigate into Fraud-Detection folder
cd /d D:\Git\Fraud-Detection

echo ===================================================
echo Working Directory: %CD%
echo ===================================================

:: Enable Git Credential Manager to use browser sign-in
echo Configuring Git Credential Manager for web login...
git config --global credential.helper manager

:: Initialize repository if needed
IF NOT EXIST ".git" (
    echo Initializing Git repository...
    git init
)

:: Stage files
echo Staging project files...
git add .

:: Commit
echo Committing files...
git commit -m "Initial commit: XGBoost Fraud Analytics complete project"

:: Set branch to main
git branch -M main

:: Configure remote URL
git remote remove origin >nul 2>&1
git remote add origin https://github.com/aadhitea/Fraud-Detection.git

echo.
echo ===================================================
echo Pushing to GitHub...
echo A browser window / pop-up will appear shorty.
echo Log in using your GitHub username and password.
echo ===================================================
echo.

:: Trigger push
git push -u origin main --force

echo.
IF %ERRORLEVEL% NEQ 0 (
    echo ===================================================
    echo PUSH FAILED! Please check the terminal error.
    echo ===================================================
) ELSE (
    echo ===================================================
    echo SUCCESS! Repository updated:
    echo https://github.com/aadhitea/Fraud-Detection
    echo ===================================================
)

pause