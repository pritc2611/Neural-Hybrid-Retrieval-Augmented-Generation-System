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

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIG
# =============================================================================
MODEL_NAME = os.getenv("EMBED_MODEL")
NUM_WORKERS = max(2, (os.cpu_count() or 4) // 2)  # half cores for embedding
BATCH_SIZE = 32  # per-worker batch

print(f"🔄  Loading model '{MODEL_NAME}' on {NUM_WORKERS} worker processes…")

# =============================================================================
# WORKER INITIALIZER  –  runs once per worker process at pool creation
# =============================================================================
_worker_model = None  # process-local model handle


def _worker_init():
    """Load the model inside each worker process exactly once."""
    global _worker_model
    from sentence_transformers import SentenceTransformer

    _worker_model = SentenceTransformer(MODEL_NAME)
    # Workers are CPU-only (GPU is shared by the main process if available)
    _worker_model = _worker_model.to("cpu")
    # Warm up
    _worker_model.encode(["warmup"], show_progress_bar=False)
    print(f"  ✅ Worker {os.getpid()} model ready")


def _worker_encode(texts: List[str]) -> List[List[float]]:
    """Called inside a worker process — model is already loaded."""
    global _worker_model
    vecs = _worker_model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vecs.tolist()


# =============================================================================
# PROCESS POOL  –  created at module load, workers initialize model
# =============================================================================
# NOTE: 'spawn' is mandatory on Windows and safer on macOS
_ctx = mp.get_context("spawn")
_pool = ProcessPoolExecutor(
    max_workers=NUM_WORKERS,
    mp_context=_ctx,
    initializer=_worker_init,
)

# Embed dim — query once in main process
from sentence_transformers import SentenceTransformer as _ST

_probe = _ST(MODEL_NAME)
EMBED_DIM = _probe.get_sentence_embedding_dimension()
del _probe
print(
    f"✅ Embedding service ready | model={MODEL_NAME} | dim={EMBED_DIM} | workers={NUM_WORKERS}"
)


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
    title="Embedding Service",
    description=f"Multiprocessing sentence-transformer ({MODEL_NAME})",
    version="4.0",
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


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "dim": EMBED_DIM,
        "workers": NUM_WORKERS,
    }


@app.post("/embed")
async def embed_query(query: Query):
    """
    Single text  → {"text": "hello"}        → {"embeddings": [...]}
    Batch texts  → {"text": ["a", "b"]}     → {"embeddings": [[...],[...]]}
    Image b64    → {"image_base64": "..."}  → zero-vector (LLM handles image)
    """
    t0 = time.time()
    loop = asyncio.get_event_loop()

    # ── Batch path ────────────────────────────────────────────────────────────
    if isinstance(query.text, list):
        texts = query.text
        embs = await loop.run_in_executor(_pool, _worker_encode, texts)
        return {
            "embeddings": embs,
            "count": len(texts),
            "dim": EMBED_DIM,
            "latency_ms": int((time.time() - t0) * 1000),
        }

    # ── Single / multimodal path ──────────────────────────────────────────────
    text_emb = None
    if query.text:
        [vec] = await loop.run_in_executor(_pool, _worker_encode, [query.text.strip()])
        text_emb = np.array(vec, dtype=np.float32)

    image_emb = None
    if query.image_base64:
        image_emb = np.zeros(EMBED_DIM, dtype=np.float32)

    if text_emb is None and image_emb is None:
        return {"error": "No text or image provided"}, 400

    if text_emb is not None and image_emb is not None:
        final = 0.6 * text_emb + 0.4 * image_emb
        n = np.linalg.norm(final)
        if n > 0:
            final /= n
    else:
        final = text_emb if text_emb is not None else image_emb

    return {
        "embeddings": final.tolist(),
        "dim": EMBED_DIM,
        "latency_ms": int((time.time() - t0) * 1000),
    }


@app.post("/embed/batch")
async def embed_batch(texts: List[str]):
    """Direct batch endpoint — same as POST /embed with list."""
    if not texts:
        return {"error": "No texts provided"}, 400
    t0 = time.time()
    loop = asyncio.get_event_loop()
    embs = await loop.run_in_executor(_pool, _worker_encode, texts)
    ms = int((time.time() - t0) * 1000)
    return {
        "embeddings": embs,
        "count": len(texts),
        "dim": EMBED_DIM,
        "latency_ms": ms,
        "avg_ms_per_text": ms / len(texts),
    }
