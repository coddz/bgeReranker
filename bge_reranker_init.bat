@echo off
REM BGE Reranker 模型下载初始化脚本
REM 自动配置环境并下载模型到指定目录

REM 配置环境变量
set HF_HOME=F:\openclaw_models\extModels
set USE_FP16=true

REM 激活conda环境
echo Activating conda environment: openclaw
call conda activate openclaw
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

REM 运行模型下载脚本
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
