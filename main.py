#!/usr/bin/env python3
"""
Windows-Safe eDNA Autoencoder Clustering API
============================================
This version completely avoids Windows threadpool issues by implementing
manual clustering without using sklearn.predict()
"""

import sys
import os

# Windows-specific fixes MUST be at the very top
if sys.platform.startswith('win'):
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from Bio import SeqIO
import re
import joblib
import pandas as pd
import numpy as np
import traceback
import tempfile
import warnings

# Suppress threading warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# TensorFlow config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

app = FastAPI(
    title="Windows-Safe eDNA API",
    description="eDNA clustering API optimized for Windows systems",
    version="1.0.0"
)

# =====================================================
# GLOBAL VARIABLES
# =====================================================
MODELS_DIR = "models"
vectorizer = None
encoder = None
kmeans = None
models_loaded = False

# =====================================================
# WINDOWS-SAFE CLUSTERING FUNCTIONS
# =====================================================
def safe_distance_calculation(embeddings, cluster_centers):
    """Calculate distances without using sklearn methods"""
    print("🔧 Using safe distance calculation...")
    
    # Manual distance calculation to avoid threadpool issues
    n_samples = embeddings.shape[0]
    n_clusters = cluster_centers.shape[0]
    
    labels = np.zeros(n_samples, dtype=int)
    
    for i, embedding in enumerate(embeddings):
        min_distance = float('inf')
        closest_cluster = 0
        
        for j, center in enumerate(cluster_centers):
            # Calculate Euclidean distance manually
            distance = np.sqrt(np.sum((embedding - center) ** 2))
            
            if distance < min_distance:
                min_distance = distance
                closest_cluster = j
        
        labels[i] = closest_cluster
        
        # Progress indicator for large datasets
        if i % 50 == 0 and i > 0:
            print(f"   Processed {i}/{n_samples} sequences...")
    
    return labels

def extract_cluster_centers(kmeans_model):
    """Safely extract cluster centers from KMeans model"""
    try:
        if hasattr(kmeans_model, 'cluster_centers_'):
            return kmeans_model.cluster_centers_
        elif hasattr(kmeans_model, '_centers'):
            return kmeans_model._centers
        else:
            # Try to find centers in model attributes
            for attr in dir(kmeans_model):
                if 'center' in attr.lower():
                    centers = getattr(kmeans_model, attr)
                    if isinstance(centers, np.ndarray) and len(centers.shape) == 2:
                        return centers
            
            raise AttributeError("Could not find cluster centers in model")
            
    except Exception as e:
        raise ValueError(f"Failed to extract cluster centers: {e}")

# =====================================================
# MODEL LOADING
# =====================================================
def load_models():
    """Load models with Windows-safe approach"""
    global vectorizer, encoder, kmeans, models_loaded
    
    print("🔧 Loading models (Windows-safe mode)...")
    print(f"📁 Looking in: {os.path.abspath(MODELS_DIR)}")
    
    if not os.path.exists(MODELS_DIR):
        print(f"❌ Models directory not found: {os.path.abspath(MODELS_DIR)}")
        return False
    
    print(f"📋 Files found: {os.listdir(MODELS_DIR)}")
    
    try:
        # Load vectorizer
        vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            print(f"❌ Vectorizer not found: {vectorizer_path}")
            return False
        
        vectorizer = joblib.load(vectorizer_path)
        print(f"✅ Vectorizer loaded (vocab: {len(vectorizer.vocabulary_)})")
        
        # Load encoder
        encoder_path = os.path.join(MODELS_DIR, "encoder.h5")
        if not os.path.exists(encoder_path):
            print(f"❌ Encoder not found: {encoder_path}")
            return False
        
        encoder = tf.keras.models.load_model(encoder_path)
        print(f"✅ Encoder loaded ({encoder.input_shape} → {encoder.output_shape})")
        
        # Load kmeans (but we won't use its predict method)
        kmeans_path = os.path.join(MODELS_DIR, "kmeans.pkl")
        if not os.path.exists(kmeans_path):
            print(f"❌ KMeans not found: {kmeans_path}")
            return False
        
        kmeans = joblib.load(kmeans_path)
        print(f"✅ KMeans loaded ({kmeans.n_clusters} clusters)")
        
        # Test cluster center extraction
        cluster_centers = extract_cluster_centers(kmeans)
        print(f"✅ Cluster centers extracted: shape {cluster_centers.shape}")
        
        models_loaded = True
        print("🚀 All models loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        traceback.print_exc()
        models_loaded = False
        return False

# =====================================================
# UTILITY FUNCTIONS
# =====================================================
def clean_sequence(seq: str) -> str:
    """Remove non-ACGT characters"""
    return re.sub(r'[^ACGT]', '', seq.upper())

def fasta_to_kmers(fasta_file: str, k: int = 6):
    """Convert FASTA to k-mers"""
    sequences = []
    seq_ids = []
    
    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = clean_sequence(str(record.seq))
        
        if len(seq) >= k:
            kmers = [seq[j:j+k] for j in range(len(seq) - k + 1)]
            kmer_string = " ".join(kmers)
            sequences.append(kmer_string)
            seq_ids.append(record.id)
    
    return sequences, seq_ids

def predict_clusters_windows_safe(fasta_file: str):
    """Windows-safe clustering prediction"""
    if not models_loaded:
        if not load_models():
            raise ValueError("Models not loaded")
    
    print("🧬 Starting Windows-safe prediction...")
    
    # Step 1: Process FASTA
    sequences, seq_ids = fasta_to_kmers(fasta_file, k=6)
    if not sequences:
        raise ValueError("No valid sequences found")
    
    print(f"📊 Processing {len(sequences)} sequences...")
    
    # Step 2: Vectorize
    X = vectorizer.transform(sequences).toarray()
    print(f"   K-mer matrix: {X.shape}")
    
    # Step 3: Generate embeddings
    embeddings = encoder.predict(X, verbose=0)
    print(f"   Embeddings: {embeddings.shape}")
    
    # Step 4: Windows-safe clustering
    cluster_centers = extract_cluster_centers(kmeans)
    labels = safe_distance_calculation(embeddings, cluster_centers)
    
    print(f"✅ Clustering complete - {len(set(labels))} clusters found")
    
    return {
        "labels": labels,
        "sequences": sequences,
        "seq_ids": seq_ids,
        "embeddings": embeddings,
        "kmer_matrix": X
    }

def calculate_diversity_metrics(labels):
    """Calculate biodiversity metrics"""
    abundance = pd.Series(labels).value_counts(normalize=True) * 100
    abundance_df = pd.DataFrame({
        "cluster": abundance.index.tolist(),
        "relative_abundance_percent": abundance.values.tolist()
    }).sort_values("cluster")
    
    richness = len(set(labels))
    proportions = abundance.values / 100
    shannon = -(proportions * np.log(proportions + 1e-10)).sum()
    simpson = 1 - np.sum(proportions ** 2)
    evenness = shannon / np.log(richness) if richness > 1 else 0
    
    return {
        "abundance_table": abundance_df.to_dict(orient="records"),
        "metrics": {
            "richness": int(richness),
            "shannon_diversity": float(shannon),
            "simpson_diversity": float(simpson),
            "evenness": float(evenness)
        }
    }

# =====================================================
# API ENDPOINTS
# =====================================================
@app.get("/")
def root():
    return {
        "message": "Windows-Safe eDNA Clustering API",
        "version": "1.0.0",
        "platform": sys.platform,
        "status": "ready" if models_loaded else "models not loaded",
        "note": "Optimized for Windows systems - avoids threadpool issues"
    }

@app.get("/health")
def health_check():
    global models_loaded
    
    if not models_loaded:
        models_loaded = load_models()
    
    if not models_loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Models not loaded"}
        )
    
    return {
        "status": "healthy",
        "platform": sys.platform,
        "models": {
            "vectorizer": vectorizer is not None,
            "encoder": encoder is not None,
            "kmeans": kmeans is not None
        }
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    """Windows-safe prediction endpoint"""
    if not models_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not file.filename.lower().endswith(('.fasta', '.fa', '.fas', '.fna')):
        raise HTTPException(status_code=400, detail="File must be FASTA format")
    
    tmp_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".fasta") as tmp_file:
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(status_code=400, detail="Empty file")
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        # Predict using Windows-safe method
        results = predict_clusters_windows_safe(tmp_path)
        diversity = calculate_diversity_metrics(results["labels"])
        
        response = {
            "success": True,
            "method": "windows_safe_clustering",
            "file_info": {
                "filename": file.filename,
                "num_sequences": len(results["sequences"]),
                "num_clusters_found": len(set(results["labels"]))
            },
            "clustering_results": {
                "cluster_labels": results["labels"].tolist(),
                "relative_abundance": diversity["abundance_table"]
            },
            "biodiversity_metrics": diversity["metrics"],
            "visualization_data": {
                "embeddings_2d": results["embeddings"][:, :2].tolist(),
                "cluster_labels": results["labels"].tolist()
            }
        }
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
    
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except:
                pass

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("🪟 Starting Windows-Safe eDNA API...")
    success = load_models()
    if not success:
        print("⚠️ Models not loaded - API will return 503 errors")
    else:
        print("✅ Windows-safe API ready!")

if __name__ == "__main__":
    import uvicorn
    print("🪟 Starting Windows-Safe eDNA API...")
    uvicorn.run("windows_safe_api:app", host="0.0.0.0", port=8000, reload=True)