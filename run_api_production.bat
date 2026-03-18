@echo off
REM BGE Reranker API Production startup script

echo.
echo ========================================
echo  BGE Reranker API Production Server  
echo ========================================
echo.

REM Production environment variables
set HOST=0.0.0.0
set PORT=8000
set MAX_DOCUMENTS=200
set MAX_DOC_LENGTH=4096
set MAX_QUERY_LENGTH=1024
set BATCH_SIZE=16
set USE_FP16=true
set ENABLE_CORS=true
set CUDA_CLEAR_CACHE=true
set KMP_DUPLICATE_LIB_OK=TRUE

REM Stability limits
set MAX_CONCURRENT_REQUESTS=8
set PROCESSING_TIMEOUT=90

REM Model cache path (override via HF_HOME if needed)
if not defined HF_HOME (
    set HF_HOME=F:\openclaw_models\extModels
)

REM Activate conda environment
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
    pause
    exit /b 1
)

call "%ACTIVATE_BAT%" openclaw

if errorlevel 1 (
    echo Error: Failed to activate conda environment 'openclaw'
    echo Please check if the environment exists and conda is properly installed
    pause
    exit /b 1
)

REM Validate dependencies
echo Checking required packages...
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Warning: PyTorch not found. Installing...
    conda install pytorch torchvision torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia
)
python -c "import FlagEmbedding" >nul 2>&1
if errorlevel 1 (
    echo Warning: FlagEmbedding not found. Installing...
    pip install FlagEmbedding
)

echo.
echo Starting Production BGE Reranker API with the following configuration:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Max Concurrent Requests: %MAX_CONCURRENT_REQUESTS%
echo   Processing Timeout: %PROCESSING_TIMEOUT%s
echo   Max Documents/Request: %MAX_DOCUMENTS%
echo   Max Doc Length: %MAX_DOC_LENGTH%
echo   Max Query Length: %MAX_QUERY_LENGTH%
echo   Batch Size: %BATCH_SIZE%
echo   Use FP16: %USE_FP16%
echo   CUDA Cache Clear: %CUDA_CLEAR_CACHE%
echo   KMP Duplicate Lib OK: %KMP_DUPLICATE_LIB_OK%
echo   Model Cache: %HF_HOME%
echo.

set HF_HOME=%HF_HOME%

REM Run production API
echo Starting Production BGE Reranker API server...
python bgeReranker_API_production.py

pause