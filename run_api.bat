@echo off
REM Batch script to run the BGE Reranker API in the openclaw conda environment

echo Ensuring conda is initialized and activating environment: openclaw
call "%USERPROFILE%\Anaconda3\Scripts\activate.bat" openclaw
if errorlevel 1 (
    call "F:\Working\anaconda3\Scripts\activate.bat" openclaw
)

if errorlevel 1 (
    echo Failed to activate conda environment. Please make sure the openclaw environment exists.
    pause
    exit /b 1
)

REM Set environment variable to handle OpenMP conflicts on some systems
set KMP_DUPLICATE_LIB_OK=TRUE

echo Checking for required dependencies...
python -c "import FlagEmbedding" > nul 2>&1
if errorlevel 1 (
    echo Installing FlagEmbedding...
    pip install FlagEmbedding
) else (
    echo FlagEmbedding is already installed
)

python -c "import fastapi" > nul 2>&1
if errorlevel 1 (
    echo Installing FastAPI and Uvicorn...
    pip install fastapi uvicorn
) else (
    echo FastAPI and Uvicorn are already installed
)

echo Starting BGE Reranker API server on port 8000...
echo Visit http://localhost:8000/docs for API documentation
python bgeReranker_API.py

pause