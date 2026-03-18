@echo off
REM Batch script to run the BGE Reranker API in the openclaw conda environment

echo Ensuring conda is initialized and activating environment: openclaw
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
echo KMP Duplicate Lib OK: %KMP_DUPLICATE_LIB_OK%
python bgeReranker_API.py

pause