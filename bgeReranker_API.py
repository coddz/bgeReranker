import os
# Set HF_HOME at the very beginning before any other imports that might use it
os.environ['HF_HOME'] = r"F:\openclaw_models\extModels"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import logging
from pathlib import Path
import torch

# Configuration constants - can be overridden by environment variables
MAX_DOCUMENTS = int(os.getenv('MAX_DOCUMENTS', '100'))  # Maximum number of documents per request
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))
ENABLE_CORS = os.getenv('ENABLE_CORS', 'true').lower() == 'true'
DEFAULT_USE_FP16 = os.getenv('USE_FP16', 'auto').lower()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="BGE Reranker API",
    description="API for BGE Reranker model (BAAI/bge-reranker-v2-m3)",
    version="1.0.0"
)

# Add CORS middleware if enabled
if ENABLE_CORS:
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


@app.on_event("startup")
def startup_event():
    """
    Load the BGE reranker model when the API starts
    """
    global model
    logger.info("Loading BGE Reranker model...")

    try:
        from FlagEmbedding import FlagReranker

        # Determine if we should use FP16
        if DEFAULT_USE_FP16 == 'auto':
            use_fp16 = torch.cuda.is_available()
        else:
            use_fp16 = DEFAULT_USE_FP16 == 'true'

        logger.info(f"CUDA available: {torch.cuda.is_available()}, using FP16: {use_fp16}")

        # Load the model
        model = FlagReranker(
            model_name_or_path="BAAI/bge-reranker-v2-m3",
            use_fp16=use_fp16
        )

        logger.info(f"Model loaded successfully! Running on device: {getattr(model, 'device', 'unknown')}")
    except ImportError:
        logger.error("FlagEmbedding is not installed. Please install it using: pip install FlagEmbedding")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
def read_root():
    return {"message": "Welcome to BGE Reranker API", "model": "BAAI/bge-reranker-v2-m3"}


@app.post("/rerank", response_model=RerankResponse)
def rerank(request: RerankRequest):
    """
    Rerank documents based on their relevance to the query.

    Args:
        request: Contains query, list of documents, optionally top_k results to return,
                and whether to normalize scores to 0-1 range

    Returns:
        Re-ranked list of documents with scores (higher = more relevant)
    """
    global model

    try:
        if not model:
            raise HTTPException(status_code=503, detail="Service unavailable: Model not loaded properly")

        if not request.documents:
            raise HTTPException(status_code=400, detail="Documents list cannot be empty")

        # Limit number of documents to prevent abuse
        if len(request.documents) > MAX_DOCUMENTS:
            raise HTTPException(
                status_code=400,
                detail=f"Too many documents: maximum allowed is {MAX_DOCUMENTS}"
            )

        # Prepare pairs of query and documents
        sentence_pairs = [[request.query, doc] for doc in request.documents]

        # Compute similarity scores
        scores = model.compute_score(sentence_pairs)

        # Handle case where there's only one score (not returned as list)
        if isinstance(scores, (int, float)):
            scores = [scores]

        # Create result pairs of index, document, and score
        results = []
        for i, (doc, score) in enumerate(zip(request.documents, scores)):
            score = float(score)
            # Normalize score to 0-1 range if requested using sigmoid
            if request.normalize_scores:
                score = 1 / (1 + 2.71828 ** (-score))

            results.append(RerankResponseItem(index=i, text=doc, score=score))

        # Sort by score in descending order
        results.sort(key=lambda x: x.score, reverse=True)

        # If top_k is specified, return only top_k results
        if request.top_k is not None and request.top_k > 0:
            results = results[:request.top_k]

        return RerankResponse(results=results)

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error during reranking: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during reranking")


@app.get("/health")
def health_check():
    """Check if the API is running and model is loaded"""
    global model
    if model is None:
        return {"status": "unhealthy", "model_loaded": False}

    cuda_available = torch.cuda.is_available()
    model_device = getattr(model, 'device', 'cpu')
    if hasattr(model_device, 'type'):
        model_device = model_device.type

    return {
        "status": "healthy",
        "model_loaded": True,
        "cuda_available": cuda_available,
        "model_device": model_device,
        "max_documents_per_request": MAX_DOCUMENTS
    }


if __name__ == "__main__":
    # Run the API server
    # Note: For production use, it's best to use a process manager or deploy with gunicorn/uvicorn with multiple workers
    print(f"Starting BGE Reranker API server on {HOST}:{PORT}...")
    print("Make sure to activate your conda environment and install dependencies:")
    print("  conda activate openclaw")
    print("  pip install -r requirements.txt")
    print(f"API docs will be available at http://localhost:{PORT}/docs")
    uvicorn.run(
        "bgeReranker_API:app",
        host=HOST,
        port=PORT,
        reload=False  # Set to True during development only
    )