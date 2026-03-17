import os
import asyncio
import math
from asyncio import Semaphore
from typing import List, Optional
import time

# Set HF_HOME at the very beginning before any other imports that might use it
if 'HF_HOME' not in os.environ:
    os.environ['HF_HOME'] = r"F:\openclaw_models\extModels"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
import uvicorn
import logging
from pathlib import Path
import torch

# Configuration constants - can be overridden by environment variables
MAX_DOCUMENTS = int(os.getenv('MAX_DOCUMENTS', '100'))  # Maximum number of documents per request
MAX_DOC_LENGTH = int(os.getenv('MAX_DOC_LENGTH', '2048'))  # 单个文档最大长度
MAX_QUERY_LENGTH = int(os.getenv('MAX_QUERY_LENGTH', '512'))  # 查询最大长度
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '32'))  # 批处理大小
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
ENABLE_CORS = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
DEFAULT_USE_FP16 = os.getenv('USE_FP16', 'auto').lower()
MAX_CONCURRENT_REQUESTS = int(os.getenv('MAX_CONCURRENT_REQUESTS', '10'))  # 最大并发请求数
PROCESSING_TIMEOUT = int(os.getenv('PROCESSING_TIMEOUT', '60'))  # 处理超时秒数

# 限制并发请求，避免CPU和GPU过载
concurrent_requests_semaphore = Semaphore(MAX_CONCURRENT_REQUESTS)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGE Reranker API Production",
    description="Production-ready API for BGE Reranker with enhanced stability and safety features (BAAI/bge-reranker-v2-m3)",
    version="1.2.0"
)

# Add CORS middleware if enabled
if ENABLE_CORS:
    # 注意：生产环境请将 allow_origins 修改为具体的域名列表，避免跨域安全风险
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Define request/response models
class QueryDocumentPair(BaseModel):
    query: str
    text: str


class RerankRequest(BaseModel):
    query: str
    documents: List[str]
    top_k: Optional[int] = None  # How many results to return (default=all)
    normalize_scores: Optional[bool] = False  # Whether to normalize scores to 0-1 range using sigmoid


class RerankResponseItem(BaseModel):
    index: int
    text: str
    score: float


class RerankResponse(BaseModel):
    results: List[RerankResponseItem]


# Model initialization will happen on startup
model = None


def batch_compute_score(model, pairs, batch_size=BATCH_SIZE):
    """
    分批处理大批量请求以避免内存问题
    """
    all_scores = []
    
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_scores = model.compute_score(batch)
        
        # 确保返回的是列表格式
        if isinstance(batch_scores, (float, int)):
            batch_scores = [batch_scores]
        
        all_scores.extend(batch_scores)
        
        # 批处理之间清理内存（可选，默认关闭，如需启用设置环境变量 CUDA_CLEAR_CACHE=true）
        if os.getenv('CUDA_CLEAR_CACHE', 'false').lower() == 'true' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return all_scores


@app.on_event("startup")
def startup_event():
    """
    Load the BGE reranker model when the API starts
    """
    global model
    logger.info("Loading BGE Reranker model...")

    try:
        from FlagEmbedding import FlagReranker
        
        # 确定是否使用 FP16
        if DEFAULT_USE_FP16 == 'auto':
            use_fp16 = torch.cuda.is_available()
        else:
            use_fp16 = DEFAULT_USE_FP16 == 'true'

        logger.info(f"CUDA available: {torch.cuda.is_available()}, using FP16: {use_fp16}")

        # 带参数优化的模型加载
        model = FlagReranker(
            model_name_or_path="BAAI/bge-reranker-v2-m3",
            use_fp16=use_fp16
        )
        
        # 预热模型（第一个请求可能较慢）
        _ = model.compute_score([["warmup query for", "warmup document"], ["testing", "verification"]])
        
        logger.info(f"Model loaded successfully! Running on device: {getattr(model, 'device', 'unknown')}")
        logger.info(f"API configuration - Max concurrent requests: {MAX_CONCURRENT_REQUESTS}, "
                    f"Processing timeout: {PROCESSING_TIMEOUT}s, "
                    f"Max documents per request: {MAX_DOCUMENTS}")

    except ImportError:
        logger.error("FlagEmbedding is not installed. Please install it using: pip install FlagEmbedding")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
def read_root():
    return {
        "message": "Welcome to Production-ready BGE Reranker API",
        "model": "BAAI/bge-reranker-v2-m3", 
        "version": "1.2.0",
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "processing_timeout_seconds": PROCESSING_TIMEOUT
    }


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """
    添加请求处理时间中间件，用于监控请求耗时
    """
    start_time = time.time()
    response = await call_next(request)  # 确保response总是被定义
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    logger.debug(f"Request {request.url.path} took {process_time:.2f}s")
    return response


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on their relevance to the query with production-grade reliability and monitoring.
    """
    global model

    # 检查系统是否准备好
    if not model:
        raise HTTPException(status_code=503, detail="Service unavailable: Model not loaded properly")

    # 尝试获取并发信号量许可，以限制并发量
    async with concurrent_requests_semaphore:
        # 检查是否请求过大或空数组，但不阻塞信号量
        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")

        # 限制文档数量
        if len(request.documents) > MAX_DOCUMENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many documents: maximum allowed is {MAX_DOCUMENTS}"
            )

        # 新增: 验证文档长度
        for i, doc in enumerate(request.documents):
            if len(doc) > MAX_DOC_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Document {i} exceeds maximum length of {MAX_DOC_LENGTH} characters"
                )

        # 验证查询长度
        if len(request.query) > MAX_QUERY_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters"
            )

        # 创建句子对
        sentence_pairs = [[request.query, doc] for doc in request.documents]
        
        try:
            # 使用超时包装处理密集计算
            scores = await asyncio.wait_for(
                asyncio.to_thread(batch_compute_score, model, sentence_pairs),
                timeout=PROCESSING_TIMEOUT
            )
        except asyncio.TimeoutError:
            logger.warning(f"Request processing timed out after {PROCESSING_TIMEOUT}s")
            raise HTTPException(
                status_code=408, 
                detail=f"Request processing timed out after {PROCESSING_TIMEOUT} seconds"
            )

        # 处理单一分数的情况
        if isinstance(scores, (int, float)):
            scores = [scores]

        # 创建包含索引、文档和分数的结果对
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            score = float(score)
            # 如果需要将分数标准化到 0-1 范围内使用 sigmoid 函数
            if request.normalize_scores:
                try:
                    score = 1 / (1 + math.exp(-score))
                except OverflowError:
                    # 处理极值情况避免数学溢出
                    score = 0.0 if score < 0 else 1.0

            results.append(RerankResponseItem(index=i, text=doc, score=score))

        # 按分数从高到低排序
        results.sort(key=lambda x: x.score, reverse=True)

        # 如果指定了 top_k，则只返回 top_k 结果
        if request.top_k is not None and request.top_k > 0:
            results = results[:request.top_k]

        logger.info(f"Processed request with {len(request.documents)} documents, returning {len(results)} results")
        return RerankResponse(results=results)


@app.get("/health")
def health_check():
    """Check if the API is running and model is loaded"""
    global model
    if model is None:
        return {
            "status": "unhealthy", 
            "model_loaded": False,
            "timestamp": time.time()
        }

    cuda_available = torch.cuda.is_available()
    model_device = getattr(model, 'device', 'cpu')
    # 获取设备类型的统一方法
    if hasattr(model_device, 'type') or (hasattr(model_device, '__class__') and 'cuda' in str(model_device.__class__).lower()):
        # 当 model_device 是 torch.device 时
        model_device = str(getattr(model_device, 'type', model_device))
    # 确保 model_device 是字符串以用于响应
    if not isinstance(model_device, str):
        model_device = str(model_device)

    return {
        "status": "healthy",
        "model_loaded": True,
        "cuda_available": cuda_available,
        "model_device": model_device,
        "max_documents_per_request": MAX_DOCUMENTS,
        "max_doc_length": MAX_DOC_LENGTH,
        "max_query_length": MAX_QUERY_LENGTH,
        "batch_size": BATCH_SIZE,
        "max_concurrent_requests": MAX_CONCURRENT_REQUESTS,
        "processing_timeout_seconds": PROCESSING_TIMEOUT,
        "current_concurrent_requests": MAX_CONCURRENT_REQUESTS - concurrent_requests_semaphore._value,
        "timestamp": time.time()
    }


@app.get("/metrics")
def metrics():
    """提供详细运行指标用于监控"""
    global model
    return {
        "active_requests": MAX_CONCURRENT_REQUESTS - concurrent_requests_semaphore._value,
        "available_slots": concurrent_requests_semaphore._value,
        "max_concurrent": MAX_CONCURRENT_REQUESTS,
        "model_ready": model is not None,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.time()
    }


if __name__ == "__main__":
    # Run the API server
    # Note: For production use, it's best to use a process manager like systemd or deployment with Gunicorn/UVicorn with appropriate worker counts
    print(f"Starting Production BGE Reranker API server on {HOST}:{PORT}...")
    print(f"Configuration: {MAX_CONCURRENT_REQUESTS} max concurrent requests, {PROCESSING_TIMEOUT}s timeout...")
    print("Make sure to activate your conda environment and install dependencies:")
    print("  conda activate openclaw")
    print("  pip install -r requirements.txt")
    print(f"API docs will be available at http://localhost:{PORT}/docs")
    print(f"Metrics endpoint: http://localhost:{PORT}/metrics")
    uvicorn.run(
        "bgeReranker_API_production:app",
        host=HOST,
        port=PORT,
        reload=False,  # 仅开发时设置为 True
        log_level="info"
    )