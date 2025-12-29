import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import time
import pickle
import io
import traceback
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI, Form, Request, UploadFile, File
import numpy as np
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import redis

load_dotenv()
# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

MAX_SHORT_MEMORY = 5       
MAX_HISTORY_CONTEXT = 4   
CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 2

HF_TOKEN = os.getenv("HF_TOKEN")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
Mongo_Url = os.getenv("MONGO_URL")
EMBED_MODEL_PATH = os.getenv("EMBEDMODEL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

all_docs = []
all_embeddings = []
image_data_store = {}


from function import   collection , query_cache  
from function import (
    get_format_text,
    get_conversation_context,
    save_message_mongo,
    save_message_redis,
    extract_and_store,
    async_model,
    lifespan,
    fit_bmencoder,
    init_storage
)

redis_clients = redis.Redis(host="redis_db", port=6379, db=0, decode_responses=True)

# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="RAG Chatbot with Memory",
    description="RAG chatbot with MongoDB + Redis conversation memory",
    version="3.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =============================================================================
# MAIN ROUTES
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = "default"

    global startup_time
    # Load history (from Redis + MongoDB)
    raw_history = get_conversation_context(session_id, include_mongo=True)

    # Convert to frontend-friendly format
    chat_history = []
    for msg in raw_history:
        chat_history.append({
            "type": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        })

    startup_time = request.app.state.startup_time
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history,
            "latency": None,
            "startup_time": f"{startup_time:.2f}s"
        }
    )

@app.post("/chat")
async def chat(request: Request):
    """Handle chat with MongoDB + Redis storage."""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        if not question:
            return JSONResponse(
                {"error": "Please enter a question"},
                status_code=400
            )
        
        request_start = time.time()
        
        # Generate response with conversation context
        answer = await async_model(question, session_id)
        
        # Convert to HTML
        answer_html = markdown.markdown(
            answer,
            extensions=[TableExtension(), FencedCodeExtension()]
        )
        
        # Calculate latency
        latency_ms = int((time.time() - request_start) * 1000)
        
        # Save to storage
        save_message_redis(session_id, "user", question)
        save_message_redis(session_id, "bot", answer)
        save_message_mongo(session_id, "user", question)
        save_message_mongo(session_id, "bot", answer_html)
        
        return JSONResponse({
            "answer": answer,
            "answer_html": answer_html,
            "latency": latency_ms,
            "session_id": session_id
        })
        
    except Exception as e:
        print("❌ /chat error:")
        traceback.print_exc()
        return JSONResponse(
            {"error": f"An error occurred: {str(e)}"},
            status_code=500
        )


# =============================================================================
# FILE UPLOAD
# =============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process documents."""
    global sparser_encode 
    
    try:
        if not (file.filename.endswith('.pdf') or file.filename.endswith('.text')):
            return JSONResponse(
                {"error": "Only PDF and TXT files are supported"},
                status_code=400
            )
        
        print(f"\n📄 Processing: {file.filename}")
        start = time.time()
        
        # Extract text
        text = extract_and_store(file.filename, file)
        print(f"✅ Extracted {len(text)} characters")
        
        # Chunk text
        chunks = get_format_text(text)
        print(f"✅ Created {len(chunks)} chunks")
        
        fit_bmencoder(chunks)
        print(f"✅ BM25 encoder updated")
    
        elapsed = time.time() - start
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks": len(chunks),
            "processing_time": f"{elapsed:.2f}s"
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# CONVERSATION HISTORY ENDPOINTS
# =============================================================================
@app.get("/history/{session_id}")
async def get_history(session_id: str, limit: int = 50):
    """Get conversation history for a session."""
    try:
        history = get_conversation_context(session_id, include_mongo=True, mongo_limit=limit)
        return JSONResponse({
            "session_id": session_id,
            "messages": history,
            "count": len(history)
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/history/{session_id}")
async def clear_session_history(session_id: str):
    """Clear history for a specific session."""
    try:
        # Clear Redis
        if redis_clients:
            redis_clients.delete(f"chat:{session_id}")
        
        # Clear MongoDB
        if collection:
            result = collection.delete_many({"session_id": session_id})
            deleted = result.deleted_count
        else:
            deleted = 0
        
        return JSONResponse({
            "message": f"Cleared history for session {session_id}",
            "deleted_messages": deleted
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================
@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "startup_time": f"{startup_time:.2f}s",
        "mongodb": collection is not None,
        "redis": redis_clients is not None,
        "query_cache_size": len(query_cache)
    }

@app.get("/stats")
async def stats():
    """Get statistics."""
    mongo_count = 0
    if collection:
        try:
            mongo_count = collection.count_documents({})
        except:
            pass
    
    return {
        "startup_time": f"{startup_time:.2f}s",
        "mongodb_messages": mongo_count,
        "redis_connected": redis_clients is not None,
        "query_cache": {
            "size": len(query_cache),
            "cached_queries": list(query_cache.keys())[:5]
        }
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear query cache."""
    query_cache.clear()
    return {"message": "Cache cleared"}
    
