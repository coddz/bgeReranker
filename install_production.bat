@echo off
REM Production installation checks and dependency setup

setlocal

echo ========================================
echo  BGE Reranker Production Setup
echo ========================================

REM Activate conda environment
echo Activating conda environment...
set "ACTIVATE_BAT="
if exist "%USERPROFILE%\Anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_BAT=%USERPROFILE%\Anaconda3\Scripts\activate.bat"
)
if not defined ACTIVATE_BAT if exist "F:\Working\anaconda3\Scripts\activate.bat" (
    set "ACTIVATE_BAT=F:\Working\anaconda3\Scripts\activate.bat"
)
if not defined ACTIVATE_BAT (
    echo Error: Could not find conda activate.bat
    echo Checked %USERPROFILE%\Anaconda3\Scripts\activate.bat
    echo Checked F:\Working\anaconda3\Scripts\activate.bat
    exit /b 1
)

call "%ACTIVATE_BAT%" openclaw
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'openclaw'
    echo Please ensure the environment exists and conda is properly installed
    exit /b 1
)

REM Set environment variable to handle OpenMP conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

REM Check and install required packages
echo.
echo Checking dependencies...

REM Check torch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch with CUDA support...
    conda install pytorch==2.8.0 torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
) else (
    echo PyTorch already installed
)

REM Check FlagEmbedding
python -c "import FlagEmbedding" >nul 2>&1
if errorlevel 1 (
    echo Installing FlagEmbedding...
    pip install FlagEmbedding
) else (
    echo FlagEmbedding already installed
)

REM Check other dependencies
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo Installing FastAPI...
    pip install fastapi>=0.104.1
)

python -c "import uvicorn" >nul 2>&1
if errorlevel 1 (
    echo Installing Uvicorn...
    pip install "uvicorn[standard]>=0.24.0"
)

REM Completion message
echo.
echo ========================================
echo  Installation completed successfully!
echo ========================================
echo.
echo KMP Duplicate Lib OK: %KMP_DUPLICATE_LIB_OK%
echo To start the PRODUCTION server:
echo   Double-click run_api_production.bat
echo.
echo To start the ENHANCED server:
echo   Double-click run_api_enhanced.bat
echo.
echo To download the model:
echo   Double-click bge_reranker_init.bat or bgeReranker_init_enhanced.py
echo.

pause