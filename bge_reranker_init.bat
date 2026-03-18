@echo off
REM BGE Reranker model initialization script
REM Configure environment and download the model to the target directory

REM Environment variables
set HF_HOME=F:\openclaw_models\extModels
set USE_FP16=true

REM Activate conda environment
echo Activating conda environment: openclaw
call "%USERPROFILE%\Anaconda3\Scripts\activate.bat" openclaw
if errorlevel 1 (
    call "F:\Working\anaconda3\Scripts\activate.bat" openclaw
)

if errorlevel 1 (
    echo ERROR: Failed to activate conda environment 'openclaw'
    echo Please make sure conda is installed and the environment exists
    pause
    exit /b 1
)

echo.
echo ==================================================
echo BGE Reranker Model Initialization
echo ==================================================
echo HF_HOME set to: %HF_HOME%
echo Using FP16: %USE_FP16%
echo.
echo This script will download the BAAI/bge-reranker-v2-m3 model
echo to the directory specified above.
echo ==================================================
echo.

pause

REM Run model download script
echo Starting model download...
python bgeReranker_init_enhanced.py

if errorlevel 1 (
    echo.
    echo ERROR: Model download failed!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ==================================================
echo ✅ Model downloaded successfully!
echo ==================================================
echo Model is cached in: %HF_HOME%
echo You can now run the API service using run_api_enhanced.bat
echo ==================================================
echo.

pause
