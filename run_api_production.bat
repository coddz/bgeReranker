@echo off
REM BGE Reranker API Production 启动脚本 - 生产级安全配置

echo.
echo ========================================
echo  BGE Reranker API Production Server  
echo ========================================
echo.

REM 设置生产环境变量
set HOST=0.0.0.0
set PORT=8000
set MAX_DOCUMENTS=200
set MAX_DOC_LENGTH=4096
set MAX_QUERY_LENGTH=1024
set BATCH_SIZE=16  REM 减少批处理大小以降低单请求内存消耗
set USE_FP16=true
set ENABLE_CORS=true
set CUDA_CLEAR_CACHE=true  REM 启用CUDA缓存清理，在持续负载下管理内存

REM 重要：限制并发和超时参数以确保稳定性
set MAX_CONCURRENT_REQUESTS=8  REM 根据服务器性能进行调整
set PROCESSING_TIMEOUT=90  REM 为长文档增加超时

REM 设置模型存储路径 (可通过 HF_HOME 自定义)
if not defined HF_HOME (
    set HF_HOME=F:\openclaw_models\extModels
)

REM 检查环境
call conda activate openclaw
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'openclaw'
    echo Please check if the environment exists and conda is properly installed
    pause
    exit /b 1
)

REM 验证依赖
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
echo   Model Cache: %HF_HOME%
echo.

REM 设置 HF_HOME
set HF_HOME=%HF_HOME%

REM 运行生产级API
echo Starting Production BGE Reranker API server...
python bgeReranker_API_production.py

pause