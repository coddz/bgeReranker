@echo off
REM 生产级安装检查和依赖设置

setlocal

echo ========================================
echo  BGE Reranker Production Setup
echo ========================================

REM 激活环境 
echo Activating conda environment...
call conda activate openclaw
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'openclaw'
    echo Please ensure the environment exists and conda is properly installed
    exit /b 1
)

REM 检查并安装所需包
echo.
echo Checking dependencies...

REM 检查 torch
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Installing PyTorch with CUDA support...
    conda install pytorch==2.8.0 torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
) else (
    echo PyTorch already installed
)

REM 检查 FlagEmbedding
python -c "import FlagEmbedding" >nul 2>&1
if errorlevel 1 (
    echo Installing FlagEmbedding...
    pip install FlagEmbedding
) else (
    echo FlagEmbedding already installed
)

REM 检查其他依赖
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

REM 完成消息
echo.
echo ========================================
echo  Installation completed successfully!
echo ========================================
echo.
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