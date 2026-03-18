@echo off
REM Installation script for FlagEmbedding in the openclaw environment

echo Activating conda environment: openclaw
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
    echo Failed to activate conda environment openclaw.
    pause
    exit /b 1
)

REM Set environment variable to handle OpenMP conflicts
set KMP_DUPLICATE_LIB_OK=TRUE

echo Installing FlagEmbedding in the openclaw environment...
echo KMP Duplicate Lib OK: %KMP_DUPLICATE_LIB_OK%
pip install FlagEmbedding

echo Installation completed. You can now run the API.
pause