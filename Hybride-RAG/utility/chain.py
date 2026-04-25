"""
chain.py — Streaming response generator and FastAPI lifespan startup.
"""
import time
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import Optional
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from fastapi import FastAPI
import requests as _req
from dotenv import load_dotenv
import os
from langsmith import traceable

load_dotenv()

LLM_NAME = os.getenv("LLM_NAME")

def _get_cfg():
    import utility.config as cfg
    return cfg


# =============================================================================
# SESSION BOOTSTRAP
# =============================================================================
@traceable(name="session_bootstrap")
async def _bootstrap_session(session_id: str):
    """Create-if-missing + return formatted conversation context."""
    from utility.storage import _create_session_sync, get_conversation_context, format_context_for_model

    t0 = time.perf_counter()
    await asyncio.to_thread(_create_session_sync, session_id)
    print(f"⏱  [_create_session_sync]      {(time.perf_counter()-t0)*1000:.1f} ms")

    t1 = time.perf_counter()
    conv_history = await get_conversation_context(session_id)
    print(f"⏱  [get_conversation_context]  {(time.perf_counter()-t1)*1000:.1f} ms")

    t2 = time.perf_counter()
    context_text = format_context_for_model(conv_history)
    print(f"⏱  [format_context_for_model]  {(time.perf_counter()-t2)*1000:.1f} ms")

    return context_text


# =============================================================================
# CACHE LOOKUP
# =============================================================================
@traceable(name="cache_lookup")
async def _cache_lookup(cache_key: str):
    cfg = _get_cfg()
    result = cfg.query_cache.get(cache_key)
    print(f"⏱  [cache_lookup] hit={result is not None}")
    return result


# =============================================================================
# PERSIST MESSAGES
# =============================================================================
@traceable(name="persist_messages")
async def _persist_messages(
    session_id: str,
    question: str,
    full_response: str,
    cache_key: Optional[str],
):
    from utility.storage import save_message_redis, _update_session_sync , save_message_mongo
    cfg = _get_cfg()

    t0 = time.perf_counter()
    save_message_redis(session_id, "user", question)
    print(f"⏱  [save_message_redis user]       {(time.perf_counter()-t0)*1000:.1f} ms")

    t1 = time.perf_counter()
    save_message_redis(session_id, "assistant", full_response)
    print(f"⏱  [save_message_redis assistant]  {(time.perf_counter()-t1)*1000:.1f} ms")

    t2 = time.perf_counter()
    save_message_mongo(session_id, "user", question)
    print(f"⏱  [save_message_mongo user]  {(time.perf_counter()-t1)*1000:.1f} ms")


    t3 = time.perf_counter()
    save_message_mongo(session_id, "assistant", full_response)
    print(f"⏱  [save_message_mongo assistant]  {(time.perf_counter()-t1)*1000:.1f} ms")


    t2 = time.perf_counter()
    await asyncio.to_thread(_update_session_sync, session_id, question, full_response[:80])
    print(f"⏱  [_update_session_sync]           {(time.perf_counter()-t2)*1000:.1f} ms")

    if cache_key:
        cfg.query_cache[cache_key] = full_response
        print(f"⏱  [cache_writed] key={cache_key[:12]}...")


# =============================================================================
# MAIN GENERATOR
# =============================================================================
@traceable(name="generate_response")
async def generate_response(
    question:     str,
    session_id:   str = "default",
    image_base64: Optional[str] = None,
):
    from utility.embedding import get_cache_key
    cfg = _get_cfg()

    t_total = time.perf_counter()
    print(f"\n{'='*60}")
    print(f"   generate_response START  session={session_id}")

    # 1. Session bootstrap
    t0 = time.perf_counter()
    context_text = await _bootstrap_session(session_id)
    print(f"⏱  [bootstrap_session TOTAL]   {(time.perf_counter()-t0)*1000:.1f} ms")

    # 2. Cache check
    cache_key = None
    if not image_base64:
        t0 = time.perf_counter()
        cache_key = get_cache_key(question, context_text, cfg.current_namespace)
        cached = await _cache_lookup(cache_key)
        print(f"⏱  [cache_key_build+lookup]    {(time.perf_counter()-t0)*1000:.1f} ms")
        if cached:
            print("   Cache HIT returning early")
            yield cached
            return

    # This number is the "dark time" you see in your traces
    pre_chain_ms = (time.perf_counter() - t_total) * 1000
    print(f"⏱  [PRE-CHAIN OVERHEAD (before LLM starts)] {pre_chain_ms:.1f} ms")

    # 3. Build chain input
    chain_input = {"question": question, "chat_history": context_text}
    if image_base64:
        chain_input["image_base64"] = image_base64

    # 4. Stream
    full_response = ""
    try:
        t_chain = time.perf_counter()
        first_chunk_logged = False

        async for chunk in cfg.final_chain.astream(chain_input):
            if not first_chunk_logged:
                print(f"⏱  [time_to_first_chunk]       {(time.perf_counter()-t_chain)*1000:.1f} ms")
                first_chunk_logged = True
            full_response += chunk
            yield chunk

        print(f"⏱  [chain TOTAL]               {(time.perf_counter()-t_chain)*1000:.1f} ms  |  chars={len(full_response)}")
        print(f"⏱  [generate_response TOTAL]   {(time.perf_counter()-t_total)*1000:.1f} ms")
        print(f"{'='*60}\n")

        # 5. Persist fire-and-forget (never blocks the last SSE chunk)
        asyncio.create_task(
            _persist_messages(session_id, question, full_response, cache_key)
        )

    except Exception as e:
        print(f"   Generation error: {e}")
        traceback.print_exc()
        yield "I encountered an error. Please try again."


# =============================================================================
# LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    from utility.storage   import init_storage
    from utility.embedding import EmbeddingClient
    from utility.retriver import HybridPineconeRetriever
    cfg = _get_cfg()

    t0 = time.time()
    print("\n" + "=" * 70)
    print("   HYBRID RAG SYSTEM STARTUP")
    print("=" * 70)

    cfg.embedding_client = EmbeddingClient(cfg.EMBED_SERVICE_URL)
    try:
        h = _req.get(f"{cfg.EMBED_SERVICE_URL}/", timeout=5)
        print(f"   Embedding service: {h.json()}")
    except Exception as e:
        print(f"   Embedding service unreachable: {e}")

    from pinecone.grpc import PineconeGRPC
    pc           = PineconeGRPC(api_key=cfg.PINECONE_KEY, pool_threads=50, timeout=10)
    cfg.pc_index = pc.Index(cfg.RAG_INDEX)
    cfg.pc_retriever = HybridPineconeRetriever(
        index=cfg.pc_index, emb_client=cfg.embedding_client,
        top_k=cfg.TOP_K_RETRIEVAL, namespace="__default__",
    )
    print("   Pinecone + Hybrid retriever ready")

    Chat_llm = ChatNVIDIA(
        model=LLM_NAME,
        api_key=cfg.NVIDIA_API_KEY, 
        temperature=0.9,
        top_p=0.95,
        max_completion_tokens=8192,)
    cfg.llm = Chat_llm
    print("LLM ready")

    from langchain_core.runnables      import RunnableParallel, RunnableLambda
    from langchain_core.output_parsers  import StrOutputParser
    from langchain_core.prompts         import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to document context.

STRICT RULES:
- Use context when relevant; otherwise don't answer this question.
- Be direct, accurate, and concise.
- COMPALSORY Cite source filenames when referencing context.
- If an image is described, incorporate it into your answer."""),
        ("user", "Context:\n{rag_context}\n\nConversation history:\n{chat_history}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    @traceable(name="async_retrieve_documents")
    async def async_retrieve(inp: dict) -> str:
        t0 = time.perf_counter()
        docs = await cfg.pc_retriever.get_relevant_documents(inp.get("question", ""))
        elapsed = (time.perf_counter() - t0) * 1000
        print(f"⏱  [async_retrieve_documents]  {elapsed:.1f} ms  |  docs={len(docs)}")
        if not docs:
            return "No relevant context found."
        context = "\n\n".join(d.page_content for d in docs)
        print(f"⏱  [context_join]              chars={len(context)}")
        return context

    cfg.final_chain = (
        RunnableParallel({
            "rag_context":  RunnableLambda(async_retrieve),
            "chat_history": lambda x: x.get("chat_history", ""),
            "question":     lambda x: x.get("question", ""),
        })
        | prompt
        | cfg.llm
        | StrOutputParser()
    )
    print("   RAG chain ready")

    app.state.startup_time = time.time() - t0
    init_storage()
    print(f"\n   SYSTEM READY in {app.state.startup_time:.2f}s\n" + "=" * 70)

    yield

    print("\n   Shutting down...")
    if cfg.embedding_client:
        await cfg.embedding_client.close()
    cfg._process_pool.shutdown(wait=False)
    cfg._thread_pool.shutdown(wait=False)