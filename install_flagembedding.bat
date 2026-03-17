@echo off
REM Installation script for FlagEmbedding in the openclaw environment

echo Activating conda environment: openclaw
call "%USERPROFILE%\Anaconda3\Scripts\activate.bat" openclaw
if errorlevel 1 (
    call "F:\Working\anaconda3\Scripts\activate.bat" openclaw
)

if errorlevel 1 (
    echo Failed to activate conda environment openclaw.
    pause
    exit /b 1
)

REM Set environment variable to handle OpenMP conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Installing FlagEmbedding in the openclaw environment...
pip install FlagEmbedding

echo Installation completed. You can now run the API.
pause