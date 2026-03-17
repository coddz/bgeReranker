#!/bin/bash

# Shell script to run the BGE Reranker API in the openclaw conda environment
# Note: This is primarily intended for Unix-like systems but keeping here for completeness

# Set environment variable to handle OpenMP conflicts 
export KMP_DUPLICATE_LIB_OK=TRUE

echo "Activating conda environment: openclaw"
if command -v conda &> /dev/null; then
    conda activate openclaw
else
    echo "Conda is not available on PATH, trying direct activation"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate openclaw
fi

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Please make sure the openclaw environment exists."
    exit 1
fi

echo "Checking for required dependencies..."
python -c "import FlagEmbedding" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing FlagEmbedding..."
    pip install FlagEmbedding
else
    echo "FlagEmbedding is already installed"
fi

python -c "import fastapi" &> /dev/null
if [ $? -ne 0 ]; then
    echo "Installing FastAPI and Uvicorn..."
    pip install fastapi uvicorn
else
    echo "FastAPI and Uvicorn are already installed"
fi

echo "Starting BGE Reranker API server on port 8000..."
echo "Visit http://localhost:8000/docs for API documentation"
python bgeReranker_API.py