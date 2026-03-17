@echo off
REM BGE Reranker API Enhanced 启动脚本

REM 设置环境变量
set HOST=0.0.0.0
set PORT=8000
set MAX_DOCUMENTS=200
set MAX_DOC_LENGTH=4096
set MAX_QUERY_LENGTH=1024
set BATCH_SIZE=32
set USE_FP16=true
set ENABLE_CORS=true

REM 激活环境
call conda activate openclaw
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'openclaw'
    echo Please check if the environment exists and conda is properly installed
    pause
    exit /b 1
)

REM 运行增强版 API
echo Starting Enhanced BGE Reranker API server...
echo Configured with:
echo   Host: %HOST%
echo   Port: %PORT%
echo   Max Documents: %MAX_DOCUMENTS%
echo   Max Doc Length: %MAX_DOC_LENGTH%
echo   Max Query Length: %MAX_QUERY_LENGTH%
echo   Batch Size: %BATCH_SIZE%
echo   Use FP16: %USE_FP16%

python bgeReranker_API_enhanced.py

pause