"""
app.py  –  Hybrid RAG API  (fixed)
Changes vs original:
  • /chat and /chat/stream no longer call save_message_redis manually
    (generate_response now handles both Redis + MongoDB saves internally)
  • /upload/multiple correctly returns combined_namespace
  • /namespaces simplified – no more broken dynamic class hack
"""

import os
import time
import traceback
import json
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from dotenv import load_dotenv

load_dotenv()

from utility.utils import (
    collection,
    query_cache,
    get_conversation_context,
    save_message_redis,
    save_message_mongo,
    extract_and_store,
    extract_and_store_multiple,
    generate_response,
    lifespan,
    current_namespace,
    _bm25_indexes,
)

app = FastAPI(
    title="Hybrid RAG System",
    description="Dense + BM25 hybrid RAG with image support",
    version="5.1",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# =============================================================================
# PAGES
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    session_id = "default"
    raw_history = await get_conversation_context(session_id, include_mongo=True)
    chat_history = [
        {
            "type": "user" if m["role"] == "user" else "assistant",
            "content": m["content"],
        }
        for m in raw_history
    ]
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "chat_history": chat_history,
            "latency": None,
            "startup_time": f"{getattr(request.app.state, 'startup_time', 0):.2f}s",
            "namespace": current_namespace,
        },
    )


# =============================================================================
# CHAT
# =============================================================================
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        image_b64 = data.get("image_base64")

        if not question:
            return JSONResponse({"error": "Please enter a question"}, status_code=400)

        print(f"\n{'='*70}\n📥 Q: {question[:100]}\n{'='*70}")
        t0 = time.time()
        full_answer = ""

        async for chunk in generate_response(
            question, session_id, image_base64=image_b64
        ):
            full_answer += chunk

        answer_html = markdown.markdown(
            full_answer,
            extensions=[TableExtension(), FencedCodeExtension()],
        )
        latency_ms = int((time.time() - t0) * 1000)
        print(f"⚡ Total: {latency_ms}ms")

        # NOTE: save_message_* is called inside generate_response
        return JSONResponse(
            {
                "answer": full_answer,
                "answer_html": answer_html,
                "latency": latency_ms,
                "session_id": session_id,
                "namespace": current_namespace,
            }
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
        question = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        image_b64 = data.get("image_base64")

        if not question:
            return JSONResponse({"error": "Please enter a question"}, status_code=400)

        async def event_gen():
            # generate_response internally saves to Redis + Mongo on completion
            async for chunk in generate_response(question, session_id, image_b64):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'namespace': current_namespace})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# UPLOAD
# =============================================================================
ALLOWED = (".pdf", ".txt", ".md")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        if not file.filename.lower().endswith(ALLOWED):
            return JSONResponse(
                {"error": f"Allowed: {', '.join(ALLOWED)}"}, status_code=400
            )
        t0 = time.time()
        chunks = await extract_and_store(file.filename, file)
        return JSONResponse(
            {
                "status": "success",
                "filename": file.filename,
                "chunks": len(chunks),
                "time": f"{time.time()-t0:.2f}s",
                "namespace": current_namespace,
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    """
    Upload multiple documents.
    All chunks go to ONE combined namespace so retrieval
    searches across all uploaded files simultaneously.
    """
    try:
        invalid = [
            f.filename for f in files if not f.filename.lower().endswith(ALLOWED)
        ]
        if invalid:
            return JSONResponse(
                {"error": f"Unsupported: {invalid}. Allowed: {', '.join(ALLOWED)}"},
                status_code=400,
            )
        t0 = time.time()
        pairs = [(f.filename, f) for f in files]
        summary = await extract_and_store_multiple(pairs)
        return JSONResponse(
            {
                "status": "success",
                "results": summary,
                "total_files": len(files),
                "time": f"{time.time()-t0:.2f}s",
                "namespace": current_namespace,  # ← combined namespace
            }
        )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# UTILITIES
# =============================================================================
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "startup_time": f"{getattr(app.state, 'startup_time', 0):.2f}s",
        "cache_size": len(query_cache),
        "namespace": current_namespace,
        "timestamp": time.time(),
    }


@app.get("/namespaces")
async def list_namespaces():
    try:
        from utility.utils import pc_index

        stats = pc_index.describe_index_stats()
        ns_list = [
            {
                "name": ns,
                "vector_count": info.vector_count,
                "bm25_docs": len(_bm25_indexes[ns].docs) if ns in _bm25_indexes else 0,
            }
            for ns, info in stats.namespaces.items()
        ]
        return {
            "namespaces": ns_list,
            "current": current_namespace,
            "total_vectors": stats.total_vector_count,
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/switch-namespace")
async def switch_namespace(request: Request):
    global current_namespace
    import utility.utils as uu

    data = await request.json()
    ns = data.get("namespace", "__default__")
    uu.current_namespace = ns
    current_namespace = ns
    return {"message": f"Switched to '{ns}'", "namespace": ns}


@app.post("/clear-cache")
async def clear_cache():
    query_cache.clear()
    return {"message": "Cache cleared", "size": 0}


@app.delete("/namespace/{namespace}")
async def delete_namespace(namespace: str):
    try:
        from utility.utils import pc_index

        pc_index.delete(delete_all=True, namespace=namespace)
        _bm25_indexes.pop(namespace, None)
        return {"message": f"Deleted namespace '{namespace}'"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# ERROR HANDLERS
# =============================================================================
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse({"error": "Not found"}, status_code=404)


@app.exception_handler(500)
async def internal_error(request, exc):
    return JSONResponse({"error": "Internal server error"}, status_code=500)
