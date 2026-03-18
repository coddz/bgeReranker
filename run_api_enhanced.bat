@echo off
REM BGE Reranker API Enhanced startup script

REM Environment variables
set HOST=0.0.0.0
set PORT=8000
set MAX_DOCUMENTS=200
set MAX_DOC_LENGTH=4096
set MAX_QUERY_LENGTH=1024
set BATCH_SIZE=32
set USE_FP16=true
set ENABLE_CORS=true
set KMP_DUPLICATE_LIB_OK=TRUE

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

REM Run enhanced API
echo Starting Enhanced BGE Reranker API server...
echo Configured with:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Max Documents: %MAX_DOCUMENTS%
echo   Max Doc Length: %MAX_DOC_LENGTH%
echo   Max Query Length: %MAX_QUERY_LENGTH%
echo   Batch Size: %BATCH_SIZE%
echo   Use FP16: %USE_FP16%
echo   KMP Duplicate Lib OK: %KMP_DUPLICATE_LIB_OK%

python bgeReranker_API_enhanced.py

pause