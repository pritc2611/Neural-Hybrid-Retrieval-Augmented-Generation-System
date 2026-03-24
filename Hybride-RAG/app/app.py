"""
app.py  –  Hybrid RAG API  v5.2
"""
import time
import traceback
import json

from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import List
import markdown
from markdown.extensions.tables import TableExtension
from markdown.extensions.fenced_code import FencedCodeExtension
from dotenv import load_dotenv

load_dotenv()

from utility.storage   import (get_full_session_history, get_all_sessions,
                                create_session, delete_session, rename_session)
from utility.ingestions import extract_and_store, extract_and_store_multiple
from utility.chain     import generate_response, lifespan

app = FastAPI(
    title="Hybrid RAG System",
    description="Dense + BM25 hybrid RAG with image support + persistent chat history",
    version="5.2",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ALLOWED = (".pdf", ".txt", ".md")


# =============================================================================
# PAGES
# =============================================================================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    import utility.config as cfg
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request":      request,
            "startup_time": f"{getattr(request.app.state, 'startup_time', 0):.2f}s",
            "namespace":    cfg.current_namespace,
        },
    )




# =============================================================================
# CHAT
# =============================================================================
@app.post("/chat")
async def chat(request: Request):
    import utility.config as cfg
    try:
        data       = await request.json()
        question   = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        image_b64  = data.get("image_base64")

        if not question:
            return JSONResponse({"error": "Please enter a question"}, status_code=400)

        t0          = time.time()
        full_answer = ""
        async for chunk in generate_response(question, session_id, image_base64=image_b64):
            full_answer += chunk

        answer_html = markdown.markdown(full_answer, extensions=[TableExtension(), FencedCodeExtension()])
        return JSONResponse({
            "answer":      full_answer,
            "answer_html": answer_html,
            "latency":     int((time.time() - t0) * 1000),
            "session_id":  session_id,
            "namespace":   cfg.current_namespace,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat/stream")
async def chat_stream(request: Request):
    try:
        data       = await request.json()
        question   = data.get("question", "").strip()
        session_id = data.get("session_id", "default")
        image_b64  = data.get("image_base64")

        if not question:
            return JSONResponse({"error": "Please enter a question"}, status_code=400)

        async def event_gen():
            import utility.config as cfg
            async for chunk in generate_response(question, session_id, image_b64):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield f"data: {json.dumps({'done': True, 'namespace': cfg.current_namespace})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# SESSIONS
# =============================================================================
@app.get("/sessions")
async def list_sessions():
    try:
        sessions = await get_all_sessions(limit=100)
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    try:
        messages = await get_full_session_history(session_id)
        return {"session_id": session_id, "messages": messages, "count": len(messages)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.delete("/sessions/{session_id}")
async def remove_session(session_id: str):
    try:
        await delete_session(session_id)
        return {"message": f"Session '{session_id}' deleted"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/sessions/{session_id}/rename")
async def rename_session_endpoint(session_id: str, request: Request):
    try:
        data      = await request.json()
        new_title = data.get("title", "").strip()
        if not new_title:
            return JSONResponse({"error": "Title required"}, status_code=400)
        await rename_session(session_id, new_title)
        return {"message": "Renamed", "session_id": session_id, "title": new_title}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/sessions/new")
async def new_session(request: Request):
    try:
        import uuid
        data = {}
        try:
            data = await request.json()
        except Exception:
            pass
        session_id = data.get("session_id") or f"session-{uuid.uuid4().hex[:12]}"
        await create_session(session_id)
        return {"session_id": session_id}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# UPLOAD
# =============================================================================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    import utility.config as cfg
    try:
        if not file.filename.lower().endswith(ALLOWED):
            return JSONResponse({"error": f"Allowed: {', '.join(ALLOWED)}"}, status_code=400)
        t0     = time.time()
        chunks = await extract_and_store(file.filename, file)
        return JSONResponse({
            "status":    "success",
            "filename":  file.filename,
            "chunks":    len(chunks),
            "time":      f"{time.time()-t0:.2f}s",
            "namespace": cfg.current_namespace,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload/multiple")
async def upload_multiple(files: List[UploadFile] = File(...)):
    import utility.config as cfg
    try:
        invalid = [f.filename for f in files if not f.filename.lower().endswith(ALLOWED)]
        if invalid:
            return JSONResponse({"error": f"Unsupported: {invalid}. Allowed: {', '.join(ALLOWED)}"}, status_code=400)
        t0      = time.time()
        summary = await extract_and_store_multiple([(f.filename, f) for f in files])
        return JSONResponse({
            "status":      "success",
            "results":     summary,
            "total_files": len(files),
            "time":        f"{time.time()-t0:.2f}s",
            "namespace":   cfg.current_namespace,
        })
    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


# =============================================================================
# UTILITIES
# =============================================================================
@app.get("/health")
async def health():
    import utility.config as cfg
    return {
        "status":       "healthy",
        "startup_time": f"{getattr(app.state, 'startup_time', 0):.2f}s",
        "cache_size":   len(cfg.query_cache),
        "namespace":    cfg.current_namespace,
        "timestamp":    time.time(),
    }


@app.get("/namespaces")
async def list_namespaces():
    import utility.config as cfg
    from utility.retriver import _bm25_indexes
    try:
        stats   = cfg.pc_index.describe_index_stats()
        ns_list = [
            {
                "name":         ns,
                "vector_count": info.vector_count,
                "bm25_docs":    len(_bm25_indexes[ns].docs) if ns in _bm25_indexes else 0,
            }
            for ns, info in stats.namespaces.items()
        ]
        return {"namespaces": ns_list, "current": cfg.current_namespace, "total_vectors": stats.total_vector_count}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/switch-namespace")
async def switch_namespace(request: Request):
    import utility.config as cfg
    data = await request.json()
    cfg.current_namespace = data.get("namespace", "__default__")
    return {"message": f"Switched to '{cfg.current_namespace}'", "namespace": cfg.current_namespace}


@app.post("/clear-cache")
async def clear_cache():
    import utility.config as cfg
    cfg.query_cache.clear()
    return {"message": "Cache cleared", "size": 0}


@app.delete("/namespace/{namespace}")
async def delete_namespace(namespace: str):
    import utility.config as cfg
    from utility.retriver import _bm25_indexes
    try:
        cfg.pc_index.delete(delete_all=True, namespace=namespace)
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