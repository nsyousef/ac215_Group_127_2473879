from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union
import numpy as np
from typing import List, Dict, Optional
import os
from contextlib import asynccontextmanager

from inference_classifier import InferenceClassifier

# Global model instance
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model
    
    checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", "/models/test_best.pth")
    device = os.getenv("DEVICE", "cpu")  # Cloud Run typically uses CPU
    
    print(f"Loading model from {checkpoint_path}...")
    model = InferenceClassifier(
        checkpoint_path=checkpoint_path,
        device=device
    )
    print("Model loaded successfully!")
    
    yield  # App runs here
    
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(title="Skin Condition Classifier API", lifespan=lifespan)


# Request/Response models
class EmbeddingInput(BaseModel):
    vision_embedding: List[float]
    text_embedding: List[float]
    top_k: Optional[int] = 5


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_idx: int
    confidence: float
    top_k: List[Dict[str, Union[str, float]]]


class TextEmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Skin Condition Classifier",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    info = model.get_model_info()
    return {
        "status": "healthy",
        "model_info": info
    }


@app.post("/embed-text", response_model=TextEmbeddingResponse)
async def embed_text(input_data: TextInput):
    """Generate text embedding from input text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        embedding = model.embed_text(input_data.text)
        return {
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: EmbeddingInput):
    """Predict skin condition from vision and text embeddings."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        vision_emb = np.array(input_data.vision_embedding, dtype=np.float32)
        text_emb = np.array(input_data.text_embedding, dtype=np.float32)
        
        result = model.predict(
            vision_embedding=vision_emb,
            text_embedding=text_emb,
            return_probs=True,
            top_k=input_data.top_k
        )
        
        # Format top_k as list of dicts
        top_k_formatted = [
            {"class": class_name, "probability": prob}
            for class_name, prob in result['top_k']
        ]
        
        return {
            "predicted_class": result['predicted_class'],
            "predicted_idx": result['predicted_idx'],
            "confidence": result['confidence'],
            "top_k": top_k_formatted
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get all available class names."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "classes": model.get_class_names(),
        "num_classes": len(model.get_class_names())
    }