"""
chain.py — Streaming response generator and FastAPI lifespan startup.
"""
import time
import traceback
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

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

@traceable(name="genrating responce")
async def generate_response(
    question:     str,
    session_id:   str = "default",
    image_base64: Optional[str] = None,
):
    from utility.storage   import (_create_session_sync, _update_session_sync,
                                   get_conversation_context, format_context_for_model,
                                   save_message_redis)
    from utility.embedding import get_cache_key
    cfg = _get_cfg()

    await asyncio.to_thread(_create_session_sync, session_id)

    conv_history = await get_conversation_context(session_id)
    context_text = format_context_for_model(conv_history)

    cache_key = None
    if not image_base64:
        cache_key = get_cache_key(question, context_text, cfg.current_namespace)
        if cache_key in cfg.query_cache:
            print("✅ Cache hit")
            yield cfg.query_cache[cache_key]
            return

    chain_input = {"question": question, "chat_history": context_text}
    if image_base64:
        chain_input["image_base64"] = image_base64

    try:
        full_response = ""
        async for chunk in cfg.final_chain.astream(chain_input):
            full_response += chunk
            yield chunk

        if cache_key:
            cfg.query_cache[cache_key] = full_response

        save_message_redis(session_id, "user",      question)
        save_message_redis(session_id, "assistant", full_response)
        await asyncio.to_thread(_update_session_sync, session_id, question, full_response[:80])

    except Exception as e:
        print(f"❌ Generation error: {e}")
        traceback.print_exc()
        yield "I encountered an error. Please try again."


@asynccontextmanager
async def lifespan(app: FastAPI):
    from utility.storage   import init_storage
    from utility.embedding import EmbeddingClient
    from utility.retriver import HybridPineconeRetriever
    cfg = _get_cfg()

    t0 = time.time()
    print("\n" + "=" * 70)
    print("🚀 HYBRID RAG SYSTEM STARTUP")
    print("=" * 70)


    cfg.embedding_client = EmbeddingClient(cfg.EMBED_SERVICE_URL)
    try:
        h = _req.get(f"{cfg.EMBED_SERVICE_URL}/health", timeout=5)
        print(f"✅ Embedding service: {h.json()}")
    except Exception as e:
        print(f"⚠️  Embedding service unreachable: {e}")

    from pinecone.grpc import PineconeGRPC
    pc           = PineconeGRPC(api_key=cfg.PINECONE_KEY, pool_threads=50, timeout=10)
    cfg.pc_index = pc.Index(cfg.RAG_INDEX)
    cfg.pc_retriever = HybridPineconeRetriever(
        index=cfg.pc_index, emb_client=cfg.embedding_client,
        top_k=cfg.TOP_K_RETRIEVAL, namespace="__default__",
    )
    print("✅ Pinecone + Hybrid retriever ready")

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    chat_llm = HuggingFaceEndpoint(
        repo_id=LLM_NAME,
        huggingfacehub_api_token=cfg.HF_TOKEN,
    )
    cfg.llm = ChatHuggingFace(llm=chat_llm)
    print("✅ LLM ready")

    from langchain_core.runnables     import RunnableParallel, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts        import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to document context.

RULES:
- Use context when relevant; otherwise don't answer this question.
- Be direct, accurate, and concise.
- Cite source filenames when referencing context.
- If an image is described, incorporate it into your answer."""),
        ("user", "Context:\n{rag_context}\n\nConversation history:\n{chat_history}\n\nQuestion: {question}\n\nAnswer:"),
    ])

    async def async_retrieve(inp: dict) -> str:
        docs = await cfg.pc_retriever.get_relevant_documents(inp.get("question", ""))
        if not docs:
            return "No relevant context found."
        return "\n\n".join(d.page_content for d in docs)

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
    print("✅ RAG chain ready")

    app.state.startup_time = time.time() - t0
    print(f"\n✅ SYSTEM READY in {app.state.startup_time:.2f}s\n" + "=" * 70)
    init_storage()

    yield

    print("\n🛑 Shutting down…")
    if cfg.embedding_client:
        await cfg.embedding_client.close()
    cfg._process_pool.shutdown(wait=False)
    cfg._thread_pool.shutdown(wait=False)
