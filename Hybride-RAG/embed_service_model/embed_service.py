"""
modelapi.py  -  High-throughput embedding service
Uses a multiprocessing-based worker pool so heavy CPU inference never
blocks the async HTTP layer.  Each worker loads the model once at
startup via initializer → zero cold-start per request.
"""

import os
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, List, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor

load_dotenv(Path(__file__).with_name("embed_service.env"))

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.getenv("EMBED_MODEL")
HF_TOKEN = os.getenv("HF_TOKEN")
NUM_WORKERS = max(2, (os.cpu_count() or 4) // 2)  # half cores for embedding
BATCH_SIZE = 32  # per-worker batch
threds = ThreadPoolExecutor(NUM_WORKERS)

print(f"🔄  Loading model '{MODEL_NAME}' on {NUM_WORKERS} worker processes…")

# =============================================================================
# WORKER INITIALIZER  –  runs once per worker process at pool creation
# =============================================================================

def _worker_encode(texts: List[str] , _worker_model) -> List[List[float]]:
    """Called inside a worker process — model is already loaded."""
    vecs = _worker_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.tolist()


# =============================================================================
# REQUEST MODEL
# =============================================================================
class Query(BaseModel):
    text: Optional[Union[str, List[str]]] = None


# =============================================================================
# FASTAPI APP
# =============================================================================
@asynccontextmanager
async def lifespan(app:FastAPI):
    global EMBED_DIM , _worker_model

    _worker_model = SentenceTransformer(MODEL_NAME,token=HF_TOKEN).to("cpu")

    from sentence_transformers import SentenceTransformer as _ST
    
    print(f"just warmmed up {len(_worker_encode('Hello',_worker_model))}")
    
    EMBED_DIM = _worker_model.get_sentence_embedding_dimension()    
    print(f"✅ Embedding service ready | model={MODEL_NAME} | dim={EMBED_DIM} | workers={NUM_WORKERS}")

    yield

    print("\n Shutting down. . . . ")


global app
app = FastAPI(
    title="Embedding Service",
    description=f"Multiprocessing sentence-transformer ({MODEL_NAME})",
    version="4.0",lifespan=lifespan
)




@app.get("/")
async def root():
    return {
        "service": "Embedding Service",
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "workers": NUM_WORKERS,
        "status": "ready",
    }


@app.post("/embed")
async def embed_query(query: Query):
    """
    Single text  → {"text": "hello"}        → {"embeddings": [...]}
    Batch texts  → {"text": ["a", "b"]}     → {"embeddings": [[...],[...]]}
    """
    t0 = time.time()
    loop = asyncio.get_event_loop()

    # ── Batch path ────────────────────────────────────────────────────────────
    if isinstance(query.text, list):
        texts = query.text
        embs = await loop.run_in_executor(threds, _worker_encode, texts , _worker_model)
        return {
            "embeddings": embs,
            "count": len(texts),
            "dim": EMBED_DIM,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # ── Single / multimodal path ──────────────────────────────────────────────
    text_emb = None
    if query.text:
        [vec] = await loop.run_in_executor(threds, _worker_encode, [query.text.strip()],_worker_model)
        text_emb = np.array(vec, dtype=np.float32)


    if text_emb is None:
        return {"error": "No text or image provided"}, 400


    return {
        "embeddings": text_emb.tolist(),
        "dim": EMBED_DIM,
        "latency_ms": int((time.time() - t0) * 1000),
    }
