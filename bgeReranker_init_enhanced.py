import os
import sys
from pathlib import Path


def download_model():
    """Download the BGE Reranker model to the specified directory."""
    # Set HF_HOME to the desired directory before importing transformers
    hf_home = os.getenv('HF_HOME', r"F:\openclaw_models\extModels")
    os.environ['HF_HOME'] = hf_home
    
    # Ensure the directory exists
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    
    print(f"Setting HF_HOME to: {hf_home}")
    print("Checking for FlagEmbedding and downloading BGE Reranker model...")
    
    try:
        from FlagEmbedding import FlagReranker
        
        print("FlagEmbedding is available. Loading model...")
        # Download the model (it will use the HF_HOME directory)
        model = FlagReranker(
            model_name_or_path="BAAI/bge-reranker-v2-m3",
            use_fp16=False  # Change to True if you want to use fp16 to speed up inference
        )
        
        print("Model loaded/downloaded successfully!")
        print(f"Model cached in: {hf_home}")
        
        # Run a simple test to verify
        print("Testing model with sample data...")
        scores = model.compute_score([["test query", "test document"], ["another query", "another doc"]])
        print(f"Sample scores: {scores}")
        
        # Clean up memory
        del model
        
    except ImportError:
        print("FlagEmbedding is not installed.", file=sys.stderr)
        print("Please install it by running:", file=sys.stderr)
        print("  pip install FlagEmbedding", file=sys.stderr)
        print("Or in your openclaw conda environment:", file=sys.stderr)
        print("  conda activate openclaw && pip install FlagEmbedding", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"An error occurred during model loading/download: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    download_model()