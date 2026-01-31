import os
import time
import traceback
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from dotenv import load_dotenv
import json

load_dotenv()

# Import optimized functions
from function import (
    collection,
    query_cache,
    get_conversation_context,
    save_message_redis,
    extract_and_store,
    generate_response,
    lifespan,
    current_namespace,
    pc_index,
    save_message_mongo
    
)



# =============================================================================
# FASTAPI APPLICATION
# =============================================================================
app = FastAPI(
    title="Production RAG System",
    description="Optimized RAG chatbot with <2s response time",
    version="4.0",
    lifespan=lifespan
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =============================================================================
# MAIN ROUTES
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with chat history"""
    session_id = "default"
    
    # Load history from Redis only (fast)
    raw_history = await get_conversation_context(session_id, include_mongo=False)
    
    chat_history = [
        {
            "type": "user" if msg["role"] == "user" else "assistant",
            "content": msg["content"]
        }
        for msg in raw_history
    ]
    
    startup_time = getattr(request.app.state, 'startup_time', 0)
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history,
            "latency": None,
            "startup_time": f"{startup_time:.2f}s",
            "namespace": current_namespace
        }
    )

@app.post("/chat")
async def chat(request: Request):
    """Handle chat requests with streaming"""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        if not question:
            return JSONResponse(
                {"error": "Please enter a question"},
                status_code=400
            )
        
        print(f"\n{'='*70}")
        print(f"📥 Question: {question[:100]}")
        print(f"📁 Namespace: {current_namespace}")
        print(f"{'='*70}\n")
        
        request_start = time.time()
        
        # Generate response (streaming)
        full_answer = ""
        async for chunk in generate_response(question, session_id):
            full_answer += chunk
        
        # Convert to HTML
        answer_html = markdown.markdown(
            full_answer,
            extensions=[TableExtension(), FencedCodeExtension()]
        )
        
        # Calculate latency
        latency_ms = int((time.time() - request_start))
        
        # Save to Redis (async, non-blocking)
        save_message_redis(session_id, "user", question)
        save_message_redis(session_id, "bot", full_answer)
        save_message_mongo(session_id, "user", question)
        save_message_mongo(session_id, "bot", full_answer)

        
        print(f"\n⚡ TOTAL LATENCY: {latency_ms}ms\n")
        
        return JSONResponse({
            "answer": full_answer,
            "answer_html": answer_html,
            "latency": latency_ms,
            "session_id": session_id,
            "namespace": current_namespace
        })
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        traceback.print_exc()
        return JSONResponse(
            {"error": f"An error occurred: {str(e)}"},
            status_code=500
        )

@app.post("/chat/stream")
async def chat_stream(request: Request):
    """Streaming endpoint for real-time responses"""
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        
        if not question:
            return JSONResponse({"error": "Please enter a question"}, status_code=400)
        
        async def event_generator():
            full_answer = ""
            
            async for chunk in generate_response(question, session_id):
                full_answer += chunk
                # Send as Server-Sent Events
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            
            # Save after complete
            save_message_redis(session_id, "user", question)
            save_message_redis(session_id, "bot", full_answer)
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        print(f"❌ Stream error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# FILE UPLOAD - OPTIMIZED
# =============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process documents"""
    try:
        allowed_extensions = ('.pdf', '.txt', '.md')
        if not file.filename.lower().endswith(allowed_extensions):
            return JSONResponse(
                {"error": f"Only {', '.join(allowed_extensions)} files are supported"},
                status_code=400
            )
        
        print(f"\n📤 Processing: {file.filename}")
        start = time.time()
        
        # Extract and store
        docs = await extract_and_store(file.filename, file)
        
        elapsed = time.time() - start
        
        print(f"✅ Upload complete: {len(docs)} chunks in {elapsed:.2f}s\n")
        
        return JSONResponse({
            "status": "success",
            "filename": file.filename,
            "chunks": len(docs),
            "processing_time": f"{elapsed:.2f}s",
            "namespace": current_namespace
        })
        
    except Exception as e:
        print(f"❌ Upload error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# UTILITY ENDPOINTS
# =============================================================================
@app.get("/health")
async def health():
    """Health check with system status"""
    return {
        "status": "healthy",
        "startup_time": f"{getattr(app.state, 'startup_time', 0):.2f}s",
        "cache_size": len(query_cache),
        "namespace": current_namespace,
        "timestamp": time.time()
    }

@app.get("/stats")
async def stats():
    """Get system statistics"""
    # Get Pinecone stats
    pinecone_stats = {}
    try:
        index_stats = pc_index.describe_index_stats()
        pinecone_stats = {
            "total_vectors": index_stats.total_vector_count,
            "dimension": index_stats.dimension,
            "namespaces": {
                ns: info.vector_count 
                for ns, info in index_stats.namespaces.items()
            }
        }
    except:
        pass
    
    return {
        "startup_time": f"{getattr(app.state, 'startup_time', 0):.2f}s",
        "cache": {
            "size": len(query_cache),
            "maxsize": query_cache.maxsize,
            "ttl": query_cache.ttl
        },
        "current_namespace": current_namespace,
        "pinecone": pinecone_stats
    }

@app.post("/clear-cache")
async def clear_cache():
    """Clear query cache"""
    query_cache.clear()
    return {"message": "Cache cleared", "size": len(query_cache)}

@app.get("/namespaces")
async def list_namespaces():
    """List available Pinecone namespaces"""
    try:
        stats = pc_index.describe_index_stats()
        namespaces = [
            {
                "name": ns,
                "vector_count": info.vector_count
            }
            for ns, info in stats.namespaces.items()
        ]
        return {
            "namespaces": namespaces,
            "current": current_namespace,
            "total_vectors": stats.total_vector_count
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/switch-namespace")
async def switch_namespace(request: Request):
    """Switch active namespace"""
    global current_namespace
    
    data = await request.json()
    namespace = data.get("namespace", "__default__")
    
    from function import current_namespace as cn
    # Update global namespace
    import function
    function.current_namespace = namespace
    current_namespace = namespace
    
    return {
        "message": f"Switched to namespace: {namespace}",
        "namespace": current_namespace
    }

@app.delete("/namespace/{namespace}")
async def delete_namespace(namespace: str):
    """Delete a namespace from Pinecone"""
    try:
        pc_index.delete(delete_all=True, namespace=namespace)
        return {
            "message": f"Deleted namespace: {namespace}",
            "success": True
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        {"error": "Not found"},
        status_code=404
    )

@app.exception_handler(500)
async def internal_error(request: Request, exc):
    return JSONResponse(
        {"error": "Internal server error"},
        status_code=500
    )
