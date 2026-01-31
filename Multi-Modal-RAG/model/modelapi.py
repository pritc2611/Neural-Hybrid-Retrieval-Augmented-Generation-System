from fastapi import FastAPI
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
import os
import torch
from typing import Optional, List, Union
import time
import numpy as np

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================


print("🔄 Loading CLIP model...")
start = time.time()

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Use GPU if available for 10x speedup
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()  # Set to evaluation mode

print(f"✅ Model loaded on {device.upper()} in {time.time() - start:.2f}s")

# =============================================================================
# REQUEST MODEL
# =============================================================================
class Query(BaseModel):
    text: Optional[Union[str, List[str]]] = None
    image_base64: Optional[str] = None

# =============================================================================
# FASTAPI APP
# =============================================================================
app = FastAPI(
    title="CLIP Embedding API",
    description="High-performance CLIP embeddings for text and images",
    version="2.0"
)

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "CLIP Embedding API",
        "model": model_path,
        "device": device,
        "status": "ready"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "device": device,
        "model_loaded": model is not None
    }

@app.post("/embed")
async def embed_query(query: Query):
    """
    Generate embeddings for text and/or images.
    
    Supports:
    - Single text: {"text": "hello"}
    - Batch text: {"text": ["hello", "world"]}
    - Image: {"image_base64": "base64_string"}
    - Multimodal: {"text": "hello", "image_base64": "..."}
    """
    
    start_time = time.time()
    embeddings = []
    
    # Process text (single or batch)
    if query.text:
        with torch.no_grad():  # Disable gradients for inference
            if isinstance(query.text, list):
                # Batch processing
                inputs = processor(
                    text=query.text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                
                text_emb = model.get_text_features(**inputs)
                text_emb = text_emb.detach().cpu()
                text_emb = np.array(text_emb)
                
                # Return list of embeddings
                return {
                    "embeddings": text_emb.tolist(),
                    "count": len(query.text),
                    "latency_ms": int((time.time() - start_time) * 1000)
                }
            else:
                # Single text
                inputs = processor(
                    text=query.text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=77
                ).to(device)
                
                text_emb = model.get_text_features(**inputs)
                text_emb = text_emb[0].detach().cpu()
                text_emb = np.array(text_emb)
                embeddings.append(text_emb)
    
    # Process image (if provided)
    if query.image_base64:
        import base64
        import io
        from PIL import Image
        
        with torch.no_grad():
            image_bytes = base64.b64decode(query.image_base64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            inputs = processor(
                images=image,
                return_tensors="pt"
            ).to(device)
            
            image_emb = model.get_image_features(**inputs)
            image_emb = image_emb[0].detach().cpu()
            image_emb = np.array(image_emb)
            embeddings.append(image_emb)
    
    # Validate
    if len(embeddings) == 0:
        return {"error": "No text or images provided"}, 400
    
    # Combine multimodal (if both provided)
    if len(embeddings) == 2:
        final_embed = 0.6 * embeddings[1] + 0.4 * embeddings[0]
    else:
        final_embed = embeddings[0]
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "embeddings": final_embed.tolist(),
        "latency_ms": latency_ms
    }

@app.post("/embed/batch")
async def embed_batch(texts: List[str]):
    """
    Batch embed multiple texts efficiently.
    
    Request: ["text1", "text2", "text3"]
    Response: [[emb1], [emb2], [emb3]]
    """
    if not texts:
        return {"error": "No texts provided"}, 400
    
    start_time = time.time()
    
    with torch.no_grad():
        inputs = processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(device)
        
        embeddings = model.get_text_features(**inputs)
        embeddings = embeddings.detach().cpu()
        embeddings = np.array(embeddings)
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return {
        "embeddings": embeddings.tolist(),
        "count": len(texts),
        "latency_ms": latency_ms,
        "avg_latency_per_text": latency_ms / len(texts)
    }
