# NeuralRAG — Hybrid Retrieval-Augmented Generation System

<div align="center">

![Version](https://img.shields.io/badge/version-5.1-7DF9C4?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-6366F1?style=flat-square)
![Status](https://img.shields.io/badge/status-production-7DF9C4?style=flat-square)

**Production-grade RAG system combining dense vector search and BM25 sparse retrieval,  
fused with Reciprocal Rank Fusion — served through a streaming FastAPI backend  
and a dark-themed glassmorphism UI.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [System Components](#system-components)
- [Technical Metrics](#technical-metrics)
- [Retrieval Pipeline](#retrieval-pipeline)
- [Document Processing Pipeline](#document-processing-pipeline)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Installation](#installation)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Frontend Features](#frontend-features)
- [Performance Design](#performance-design)
- [Storage Layer](#storage-layer)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Overview

NeuralRAG is a fully async, production-ready RAG system built for speed and retrieval quality. It addresses the core weakness of pure vector-only RAG — poor exact-match recall — by running BM25 sparse retrieval in parallel with dense semantic search and fusing both result sets using Reciprocal Rank Fusion (RRF).

The system is designed around three principles:

**1. Hybrid always beats single-modal retrieval.**  
Dense search finds semantically similar content that may not share keywords. BM25 finds exact-match content that may not be semantically similar. RRF fusion captures both, consistently outperforming either alone.

**2. Latency should scale with concurrency, not document count.**  
All embedding batches are dispatched concurrently via `asyncio.gather`. CPU-bound extraction runs in a `ProcessPoolExecutor`. I/O-bound operations run in a `ThreadPoolExecutor`. The event loop is never blocked.

**3. Multi-file uploads belong in a shared namespace.**  
When multiple documents are uploaded together, all chunks land in a single merged Pinecone namespace and a single BM25 index, so retrieval searches across all documents simultaneously without namespace-switching.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser / Client                        │
│              Dark UI · SSE Streaming · Namespace Switcher       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP / SSE
┌────────────────────────────▼────────────────────────────────────┐
│                    RAG App  (app.py · port 7000)                │
│              FastAPI · Jinja2 · Static Files                    │
└──────┬──────────────────────────────────────┬───────────────────┘
       │ asyncio.gather                       │ LangChain Chain
┌──────▼──────────┐                 ┌─────────▼─────────────────┐
│  Embedding Svc  │                 │     LLM  (Gemini)         │
│  modelapi.py    │                 │  gemini-2.5-flash-lite    │
│  port 8001      │                 │  streaming=True           │
│                 │                 └───────────────────────────┘
│  ProcessPool    │
│  N workers      │     ┌──────────────────────────────────────┐
│  e5-base-v2     │     │          Retrieval Layer             │
└──────┬──────────┘     │                                      │
       │                │  ┌─────────────┐  ┌───────────────┐  │
       │ HTTP /embed    │  │ Dense Search│  │  BM25 Search  │  │
       │                │  │  Pinecone   │  │  In-Memory    │  │
       │                │  │  gRPC       │  │  Per-NS Index │  │
       │                │  └──────┬──────┘  └──────┬────────┘  │
       │                │         │    asyncio      │           │
       │                │         └──────┬──────────┘           │
       │                │                │ RRF Fusion           │
       │                │         Top-K Documents               │
       │                └──────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────┐
│              Storage Layer              │
│  ┌──────────┐  ┌───────┐  ┌─────────┐   │
│  │ Pinecone │  │ Redis │  │ MongoDB │   │
│  │ Vectors  │  │ Cache │  │ History │   │
│  └──────────┘  └───────┘  └─────────┘   │
└─────────────────────────────────────────┘
```

---

## System Components

### `embed_service.py` — Embedding Service

A dedicated FastAPI microservice that hosts the sentence-transformer model. Runs on a separate port from the main RAG application so model inference never competes with request handling.

| Property | Value |
|---|---|
| Model | `intfloat/e5-base-v2` (configurable via `EMBED_MODEL`) |
| Embedding dimension | 768 |
| Normalization | L2-normalized (cosine similarity ready) |
| Inference backend | `ProcessPoolExecutor`, `cpu_count // 2` workers |
| Worker initialization | Each worker loads the model **once** via `initializer` |
| Batch size per worker | 32 texts |
| Multiprocessing context | `spawn` (safe on Windows and macOS) |
| Endpoints | `GET /health`, `POST /embed`, `POST /embed/batch` |

**Why a process pool?** Python's GIL prevents true CPU parallelism in threads. By using `ProcessPoolExecutor` with `spawn` context and a per-worker model initializer, every CPU core gets a fully independent model instance. Concurrent embedding requests are dispatched across cores with zero GIL contention.

---

### `utility/utils.py` — Core RAG Engine

The heart of the system. 718 lines covering retrieval, embedding, document processing, storage, caching, and the LangChain chain assembly.

**Key classes:**

| Class | Responsibility |
|---|---|
| `EmbeddingClient` | Async HTTP client to embedding service with connection pooling, TTL caching, and 120s timeout |
| `BM25Index` | In-memory BM25 implementation (k1=1.5, b=0.75) with incremental `build()` and per-namespace isolation |
| `HybridPineconeRetriever` | Runs dense + sparse search concurrently via `asyncio.gather`, applies RRF, deduplicates results |

**Concurrency model:**

| Pool | Workers | Used for |
|---|---|---|
| `ProcessPoolExecutor` | `max(2, cpu_count − 1)` | PDF extraction, TXT/MD extraction (CPU-bound) |
| `ThreadPoolExecutor` | 16 | BM25 index updates, MongoDB saves (I/O-bound) |
| `asyncio.gather` | — | Parallel embedding batches, parallel dense+sparse search |

---

### `app/app.py` — RAG API

FastAPI application with 10 endpoints. Handles document upload, chat, streaming, namespace management, and health checks.

**Namespace strategy for multi-file uploads:**  
When multiple files are uploaded together, the system builds a *combined namespace* from all filenames (e.g., `doc_a__doc_b__report_c`) and upserts all chunks there. This means retrieval searches across all files simultaneously without any namespace-switching logic at query time.

---

## Technical Metrics

### Codebase

| File | Lines | Purpose |
|---|---|---|
| `utility/utils.py` | 718 | Core engine: retrieval, embedding, storage, chain |
| `static/style.css` | 648 | UI design system, 26 CSS custom properties |
| `static/script.js` | 518 | Frontend controller, all 57 HTML IDs wired |
| `app.py` | 268 | FastAPI routes, 10 endpoints |
| `templates/index.html` | 338 | Jinja2 template, 57 element IDs |
| `modelapi.py` | 178 | Embedding microservice |
| **Total** | **2,668** | |

### Retrieval Configuration

| Parameter | Value | Notes |
|---|---|---|
| `CHUNK_SIZE` | 512 tokens | `RecursiveCharacterTextSplitter` |
| `CHUNK_OVERLAP` | 100 tokens | Prevents context loss at boundaries |
| `TOP_K_RETRIEVAL` | 5 | Final fused results passed to LLM |
| `DENSE_WEIGHT` | 0.6 | RRF dense contribution |
| `SPARSE_WEIGHT` | 0.4 | RRF BM25 contribution |
| `RRF k` | 60 | Standard RRF damping constant |
| Dense candidate pool | 10 | `top_k * 2` before fusion |
| Sparse candidate pool | 10 | `top_k * 2` before fusion |

### Caching

| Cache | Max items | TTL | Scope |
|---|---|---|---|
| `query_cache` | 2,000 | 30 min | Full query responses |
| `embedding_cache` | 5,000 | 60 min | Single-text embeddings |

### Embedding Service

| Parameter | Value |
|---|---|
| Model | `intfloat/e5-base-v2` |
| Dimensions | 768 |
| Batch size per request | 32 |
| Client timeout | 120 seconds |
| Connection pool (total) | 200 connections |
| Connection pool (per host) | 100 connections |
| Keepalive | 60 seconds |

### Storage

| Store | Purpose | TTL / Limit |
|---|---|---|
| Redis | Short-term conversation cache | 24h, last 10 messages |
| MongoDB | Long-term conversation history | Persistent, last 30 fetched |
| Pinecone (gRPC) | Dense vector index | Persistent, `pool_threads=50` |
| BM25 in-memory | Sparse keyword index | Per-namespace, per-process lifetime |

### Conversation Memory

| Parameter | Value |
|---|---|
| `MAX_SHORT_MEMORY` | 10 messages (Redis) |
| `MAX_HISTORY_CONTEXT` | 6 messages injected into prompt |
| History source | Redis first → MongoDB fallback (deduplicated) |
| Context format | Last 4 messages, 300 chars each |

---

## Retrieval Pipeline

```
User query
    │
    ├──────────────────────────────────────────────┐
    │                                              │
    ▼                                              ▼
Dense Search (async)                    Sparse Search (async)
    │                                              │
    │  1. Embed query → 768-dim vector             │  1. Tokenize query
    │  2. Pinecone gRPC query                      │  2. BM25 score all docs
    │     top_k=10, include_metadata               │     in namespace
    │  3. Return (id, score, metadata)             │  3. Return (idx, score)
    │                                              │
    └──────────────────┬───────────────────────────┘
                       │ asyncio.gather (concurrent)
                       ▼
              RRF Fusion
                       │
                       │  score(doc) = 0.6 / (60 + dense_rank)
                       │            + 0.4 / (60 + sparse_rank)
                       │
                       ▼
              Top-5 fused results
              (deduplicated by text content)
                       │
                       ▼
              LangChain RAG Chain
                       │
                       │  Parallel via RunnableParallel:
                       │  ├── rag_context  ← retrieved docs
                       │  ├── chat_history ← last 6 messages
                       │  └── question     ← user input
                       │
                       ▼
              Gemini 2.5 Flash Lite
              (streaming=True)
                       │
                       ▼
              SSE stream → Browser
```

**RRF formula:**

```
score(d) = Σ  weight_i / (k + rank_i(d))
```

Where `k = 60` (standard damping constant that reduces impact of rank differences at the top), `weight_dense = 0.6`, `weight_sparse = 0.4`.

---

## Document Processing Pipeline

```
File upload (PDF / TXT / MD)
    │
    ▼
ProcessPoolExecutor worker
    │
    ├── PDF → PyMuPDF text extraction
    │         page-by-page, joined with \n\n
    │
    └── TXT/MD → direct string read
                 (also in ProcessPool, non-blocking)
    │
    ▼
RecursiveCharacterTextSplitter
    chunk_size=512, overlap=100
    separators: ["\n\n", "\n", ". ", " ", ""]
    │
    ▼
Concurrent embedding (asyncio.gather)
    │
    ├── Batch 0 (32 chunks) ──► /embed → 768-dim vectors
    ├── Batch 1 (32 chunks) ──► /embed → 768-dim vectors
    └── Batch N (32 chunks) ──► /embed → 768-dim vectors
    │   (all dispatched simultaneously)
    │
    ▼
Pinecone upsert (batches of 100, concurrent)
    │   vector ID format: {namespace}-{source}-{chunk_id}
    │   metadata: { text, source, chunk_id }
    │
    ▼
BM25 index update (ThreadPoolExecutor)
    │   incremental: existing docs + new docs
    │   rebuilds IDF and TF tables
    │
    ▼
Namespace updated → current_namespace
```

**Multi-file upload:**

```
[file_a.pdf, file_b.md, file_c.txt]
    │
    │  combined namespace = "file_a__file_b__file_c"
    │
    ├── asyncio.create_task(extract_and_store(file_a, ns=combined))
    ├── asyncio.create_task(extract_and_store(file_b, ns=combined))
    └── asyncio.create_task(extract_and_store(file_c, ns=combined))
    │
    │  asyncio.gather(*tasks)  ← all three run in parallel
    │
    ▼
  Summary: { filename: { status, chunks } }
```

---

## API Reference

### RAG Application (port 8000)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve chat UI (Jinja2 template) |
| `POST` | `/chat` | Non-streaming chat with optional image |
| `POST` | `/chat/stream` | SSE streaming chat with optional image |
| `POST` | `/upload` | Upload a single document |
| `POST` | `/upload/multiple` | Upload multiple documents → merged namespace |
| `GET` | `/health` | System health, cache size, startup time |
| `GET` | `/namespaces` | List all Pinecone namespaces with vector + BM25 counts |
| `POST` | `/clear-cache` | Clear query response cache |
| `DELETE` | `/namespace/{namespace}` | Delete namespace from Pinecone + BM25 |

**Chat request body:**
```json
{
  "question": "string",
  "session_id": "string",
  "image_base64": "string | null",
  "mode": "hybrid | dense | sparse"
}
```

**Upload multiple response:**
```json
{
  "status": "success",
  "results": {
    "doc_a.pdf": { "status": "ok", "chunks": 37 },
    "doc_b.md":  { "status": "ok", "chunks": 19 }
  },
  "total_files": 2,
  "time": "14.3s",
  "namespace": "doc_a__doc_b"
}
```

### Embedding Service (port 8001)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Model status, dimension, worker count |
| `POST` | `/embed` | Embed single text or batch (list of strings) |
| `POST` | `/embed/batch` | Explicit batch endpoint |

**Embed request:**
```json
{ "text": "single string" }
{ "text": ["batch", "of", "strings"] }
{ "image_base64": "..." }
```

---

## Configuration

All tunable parameters are defined at the top of `utility/utils.py`:

```python
# Memory
MAX_SHORT_MEMORY     = 10    # messages kept in Redis
MAX_HISTORY_CONTEXT  = 6     # messages injected into each prompt

# Chunking
CHUNK_SIZE           = 512   # tokens per chunk
CHUNK_OVERLAP        = 100   # token overlap between chunks

# Retrieval
TOP_K_RETRIEVAL      = 5     # final results after RRF fusion
EMBEDDING_BATCH_SIZE = 32    # texts per embedding request

# RRF weights (must sum to 1.0)
DENSE_WEIGHT         = 0.6
SPARSE_WEIGHT        = 0.4
```

---

## Installation

### Prerequisites

- Python 3.10+
- A Pinecone account with a 768-dimension index
- A Huggingface Token key
- Redis instance
- MongoDB Atlas or self-hosted instance

### 1. Clone and install dependencies

```bash
git clone https://github.com/pritc2611/Neural-Hybrid-Retrieval-Augmented-Generation-System
cd neural-rag

pip install -r requirements.txt

cd embed_service_model
pip install -r requirements.txt

```

### 2. Set environment variables

```bash
cp .env.example .env
# Edit .env with your credentials (see Environment Variables section)
```

### 3. Start the embedding service

```bash
uvicorn modelapi:app --host 0.0.0.0 --port 7000 --workers 1
```

> **Note:** Use `--workers 1` — the embedding service manages its own internal `ProcessPoolExecutor`. Multiple uvicorn workers would each spawn their own pool, multiplying memory usage.

### 4. Start the RAG application

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open the UI

Navigate to `http://localhost:7000`

---

> **Redis port:** hardcoded to `17564` in `utils.py`. Change `init_storage()` if your instance uses the default `6379`.

---

## Project Structure

```
neural-rag/
│
├── app/  
|    └── app.py
|                           # FastAPI application
├── embed_service_model/
|     └──embed_service.py   # Embedding microservice (ProcessPool)
│
├── utility/
│   └── utils.py            # Core engine: retrieval, chains, storage
│
├── templates/
│   └── index.html          # Jinja2 chat UI template (57 element IDs)
│
├── static/
│   ├── style.css           # Design system (26 CSS vars, dark/light themes)
│   └── script.js           # Frontend controller (all IDs wired, SSE)
│
├── .env                    # Environment variables (not committed)
├── .env.example            # Template with all required keys
└── README.md               # This file
```

---

## Frontend Features

The UI is a single-page application served by Jinja2. It communicates with the backend exclusively via `fetch` and SSE.

### Design System

| Property | Value |
|---|---|
| Fonts | Syne (headings), Inter (body), JetBrains Mono (code/mono) |
| Themes | Dark (default) + Light (localStorage persistence) |
| CSS custom properties | 26 variables |
| Animations | `fade-up`, `slide-in`, `orb-drift`, `bounce` (typing), `shimmer` (progress) |
| Layout | Sidebar + main split, fully responsive |

### Chat Features

- **SSE streaming** — tokens appear as they generate; typing indicator stays visible until the first chunk arrives (no blank bubble flash)
- **Typing stage indicator** — cycles through 5 status messages during retrieval and generation
- **Markdown rendering** — `marked.js` with `highlight.js` (Tokyo Night Dark theme)  
- **Image attachment** — base64-encoded, previewed inline, sent to Gemini for multimodal reasoning
- **Message actions** — copy, thumbs up / thumbs down per message
- **Auto-scroll lock** — stops auto-scrolling if the user scrolls up to read
- **Send button state** — disabled until text is typed or image is attached

### Upload Features

- **Drag-and-drop** or click-to-browse file selection
- **Multi-file queue** with per-file remove buttons
- **Cancel button** to clear the queue
- **Per-file status rows** — show ⏳ / ✅ / ❌ with chunk counts as each file completes
- **Progress bar** with animated shimmer and live `%` counter
- **Merged namespace display** — badge updates to the combined namespace after upload

### Sidebar Features

- **Namespace switcher** — click the namespace badge to open a dropdown of all Pinecone namespaces with vector counts and BM25 doc counts; click to switch
- **Refresh button** — re-fetches namespace list from `/namespaces`
- **Retrieval mode pills** — Hybrid / Dense / BM25; updates topbar label and description text
- **Live stats chips** — message count, system boot time, last response latency (ms), total chunks stored
- **Session history** — last 8 queries listed, click to re-ask
- **Status indicator** — green dot (idle) / amber dot (processing)

---

## Solve Problems (Design The Performance)

### Why ProcessPoolExecutor for extraction?

`RecursiveCharacterTextSplitter` and PyMuPDF are CPU-bound. Running them in the async event loop would block all other requests. Running them in a `ThreadPoolExecutor` gives no benefit because of the GIL. `ProcessPoolExecutor` gives true parallelism — a large `.md` file that previously blocked the event loop for ~52 seconds now completes in under 2 seconds.

### Why concurrent embedding batches?

The original implementation embedded batches sequentially:

```python
# Before: sequential — O(n_batches) latency
for batch in batches:
    embs = await embed(batch)  # wait for each one

# After: concurrent — O(1) latency regardless of batch count
results = await asyncio.gather(*[embed(b) for b in batches])
```

A 708-chunk document previously required 11 sequential HTTP calls (~8 seconds each = ~88 seconds). With `asyncio.gather`, all batches are in-flight simultaneously — limited only by the embedding service's throughput.

### Why gRPC for Pinecone?

`pinecone-client[grpc]` uses HTTP/2 multiplexing and binary serialization. For vector upserts and queries, this is measurably faster than the REST client, especially when batching 100-vector upserts concurrently.

### Why separate embedding service?

The embedding model (~270MB for e5-base-v2) is loaded once per process, shared across all requests. Running it in the same process as FastAPI would block startup and compete for memory. The separate service also allows independent scaling — you can run multiple embedding service replicas behind a load balancer without touching the RAG application.

---

## Storage Layer

### Redis — Short-term conversation cache

```
Key:   chat:{session_id}
Type:  List (Redis)
TTL:   86400 seconds (24 hours)
Limit: Last 10 messages (ltrim)
```

Redis is the fast path for conversation context. Every message is appended and the list is trimmed to `MAX_SHORT_MEMORY = 10` on every write.

### MongoDB — Long-term conversation history

```
Database:   chat-collection
Collection: conversations
Index:      (session_id ASC, timestamp DESC)
Schema:     { session_id, role, content, timestamp }
```

MongoDB stores the full conversation history. `get_conversation_context()` reads Redis first, then fetches older messages from MongoDB (up to 30), deduplicates, and returns the last 6 for prompt injection. Writes to MongoDB are fire-and-forget via `run_in_executor` — they never add latency to responses.

### Pinecone — Dense vector index

```
Dimensions:   768
Metric:       Cosine (L2-normalized vectors)
Transport:    gRPC (pool_threads=50)
Upsert batch: 100 vectors per call
Namespace:    per-document or merged multi-doc
```

### BM25 — In-memory sparse index

The BM25 index is a pure Python implementation with no external dependencies. It lives entirely in memory, meaning:

- Query latency is microseconds (no network I/O)
- It resets on process restart (documents must be re-uploaded or persisted separately)
- Each namespace gets an independent `BM25Index` instance in the `_bm25_indexes` dict

---

## Known Limitations

| Limitation | Details |
|---|---|
| BM25 not persistent | The in-memory BM25 index is lost on process restart. Documents stored in Pinecone remain, but BM25 must be rebuilt by re-uploading or by loading from a snapshot on startup. |
| Single-process BM25 | The BM25 index lives in the main process. Multiple uvicorn workers would each have an independent, potentially out-of-sync index. Run with `--workers 1` unless BM25 persistence is added. |
| Image embeddings | Image input is passed to Gemini for understanding but is **not** embedded into Pinecone. Image content does not contribute to retrieval — only the text question does. |
| Namespace max length | Pinecone namespace names are capped at 63 characters. The merged namespace builder truncates each filename stem to 20 characters before joining. |
| Redis port | Hardcoded to `17564`. Change `init_storage()` if your Redis uses port `6379`. |



<div align="center">

Built with FastAPI · Pinecone · Huggingface-endpoint · sentence-transformers · LangChain

</div>
