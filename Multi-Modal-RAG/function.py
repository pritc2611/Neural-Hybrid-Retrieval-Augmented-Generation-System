import os
import time
import traceback
import json
import re
from contextlib import asynccontextmanager
from fastapi import FastAPI
import numpy as np
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from collections import OrderedDict
from cachetools import TTLCache
import hashlib
import asyncio
import aiohttp
from typing import List, Optional

load_dotenv()

# =============================================================================
# CONFIGURATION 
# =============================================================================
MAX_SHORT_MEMORY = 5
MAX_HISTORY_CONTEXT = 4
CHUNK_SIZE = 600 
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 3  
EMBEDDING_BATCH_SIZE = 64 

# Environment variables
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
Mongo_Url = os.getenv("MONGO_URL")
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PASS = os.getenv("REDIS_PASS")

# Global state
temp_docs = []
temp_embeddings = []
db = None
collection = None
redis_client = None
current_namespace = "__default__" 

# =============================================================================
# OPTIMIZED CACHING
# =============================================================================
query_cache = TTLCache(maxsize=2000, ttl=1800)  # 30min TTL, 2000 items
embedding_cache = TTLCache(maxsize=5000, ttl=3600)  # Cache embeddings

def get_cache_key(question: str, context: str, namespace: str = "__default__") -> str:
    """Generate cache key with namespace awareness"""
    context_hash = hashlib.md5(context.encode()).hexdigest()[:8]
    combined = f"{question.strip().lower()}|{context_hash}|{namespace}"
    return hashlib.sha256(combined.encode()).hexdigest()[:24]

# =============================================================================
# DATABASE INITIALIZATION - CONNECTION POOLING
# =============================================================================
def init_storage():
    """Initialize MongoDB and Redis Storages"""
    global mongo_client, db, collection, redis_client
    
    try:
        from pymongo import MongoClient, ASCENDING, DESCENDING 
        import redis
        from pymongo.server_api import ServerApi
        # MongoDB with aggressive connection pooling

        
# Create a new client and connect to the server
        mongo_client = MongoClient(Mongo_Url,server_api=ServerApi("1"),)
        db = mongo_client["chat-colleaction"]
        collection = db["user-boat-coll"]
        
        # Create indexes (non-blocking)
        try:
            collection.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="session_timestamp_idx",
                background=True
            )
        except:
            pass
        
        print("✅ MongoDB connected")
        
        # Redis with connection pooling
        
        redis_client = redis.Redis(
                host=REDIS_HOST,
                port=10405,
                decode_responses=True,
                username="default",
                password=REDIS_PASS,)
        
        redis_client.ping()
        print("✅ Redis connected")
        
    except Exception as e:
        print(f"⚠️ Storage init failed: {e}")
        collection = None
        redis_client = None

# =============================================================================
# STORAGE OPERATIONS
# =============================================================================
def save_message_redis(session_id: str, role: str, content: str):
    """Non-blocking Redis save"""
    if redis_client is None:
        return
    
    try:
        key = f"chat:{session_id}"
        msg = json.dumps({"role": role, "content": content[:500]})  # Limit size
        redis_client.rpush(key, msg)
        redis_client.ltrim(key, -MAX_SHORT_MEMORY, -1)
        redis_client.expire(key, 86400)  # 24h expiry
    except:
        pass 

def save_message_mongo(session_id: str, role: str, content: str):
    """Save message to MongoDB."""
    if collection is not None:
        try:
            collection.insert_one({
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": time.time()
            })
        except Exception as e:
            print(f"MongoDB save failed: {e}")


def get_short_memory(session_id: str) -> list:
    """Get recent messages from Redis"""
    if redis_client is None:
        return []
    
    try:
        key = f"chat:{session_id}"
        raw_msgs = redis_client.lrange(key, 0, -1)
        return [json.loads(m) for m in raw_msgs]
    except:
        return []

async def get_conversation_context(session_id: str, include_mongo: bool = False) -> list:
    """Optimized async context retrieval"""
    # Fast path: Redis only
    context = await asyncio.to_thread(get_short_memory, session_id)
    
    if not include_mongo or collection is None:
        return context[-MAX_HISTORY_CONTEXT:]
    
    # Slow path: Include MongoDB
    try:
        mongo_msgs = await asyncio.to_thread(
            lambda: list(collection.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1}
            ).sort("timestamp", -1).limit(20))
        )
        
        mongo_msgs.reverse()
        redis_content = {(m['role'], m['content']) for m in context}
        
        for msg in mongo_msgs:
            if (msg['role'], msg['content']) not in redis_content:
                context.insert(0, msg)
                
    except:
        pass
    
    return context[-MAX_HISTORY_CONTEXT:]

def format_context_for_model(messages: list) -> str:
    """Format conversation history efficiently"""
    if not messages:
        return ""
    
    parts = [
        f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content'][:200]}"
        for msg in messages[-3:]  # Only last 3 exchanges
    ]
    
    return "\n".join(parts) if parts else ""

# =============================================================================
# EMBEDDING SERVICE - ASYNC WITH CONNECTION POOLING
# =============================================================================
class EmbeddingClient:
    """embedding client with connection pooling"""

    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")
        self.session = None
        self.timeout = aiohttp.ClientTimeout(total=10, connect=2)

    async def _get_session(self):
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=50,
                ttl_dns_cache=300,
                keepalive_timeout=30
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout
            )
        return self.session

    async def embed(self, text: Optional[str] = None, texts: Optional[List[str]] = None):
        if text and texts:
            raise ValueError("Provide either text or texts, not both")
        if not text and not texts:
            raise ValueError("Must provide text or texts")

        cache_key = None
        if text:
            cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
            if cache_key in embedding_cache:
                return embedding_cache[cache_key]

        payload = {"text": texts if texts else text}
        session = await self._get_session()

        try:
            async with session.post("http://embedding_service:8000/embed", json=payload) as response:
                response.raise_for_status()
                result = await response.json()

                if "embeddings" not in result:
                    raise RuntimeError(f"Invalid embedding response: {result}")

                embeddings = result["embeddings"]

                if cache_key:
                    embedding_cache[cache_key] = embeddings

                return embeddings

        except asyncio.TimeoutError:
            print("⚠️ Embedding timeout")
            if texts:
                return [[0.0] * 512 for _ in texts]
            return [0.0] * 512

        except Exception as e:
            print(f"⚠️ Embedding error: {e}")
            if texts:
                return [[0.0] * 512 for _ in texts]
            return [0.0] * 512

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

# Global embedding client
embedding_client = None

# =============================================================================
# DOCUMENT PROCESSING - OPTIMIZED FOR TEXT-ONLY
# =============================================================================
async def extract_and_store(filename: str, data):
    """Extract text from PDF and store in Pinecone"""
    global current_namespace, temp_docs, temp_embeddings
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    import pymupdf
    
    temp_docs.clear()
    temp_embeddings.clear()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    print(f"📄 Extracting from: {filename}")
    start_time = time.time()
    
    if filename.lower().endswith(".pdf"):
        pdf_bytes = data.file.read()
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        # Extract all text efficiently
        all_text = []
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():
                all_text.append(text)
        
        # Combine and split
        full_text = "\n\n".join(all_text)
        chunks = splitter.split_text(full_text)
        
        
        # Create documents
        for i, chunk in enumerate(chunks):
            temp_docs.append(Document(
                page_content=chunk,
                metadata={"chunk_id": i, "type": "text", "source": filename}
            ))
        
        # Batch embed all chunks
        texts = [doc.page_content for doc in temp_docs]
        
        # Embed in batches
        batch_size = EMBEDDING_BATCH_SIZE
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = await embedding_client.embed(texts=batch)
            
            # Handle batch response
            if isinstance(embeddings[0], list):
                all_embeddings.extend(embeddings)
            else:
                all_embeddings.append(embeddings)
        
        temp_embeddings = all_embeddings
        print(f"✅ Extracted {len(chunks)} chunks from {len(doc)} pages")
        
    elif filename.lower().endswith((".txt", ".md")):
        text = data.file.read().decode("utf-8")
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            temp_docs.append(Document(
                page_content=chunk,
                metadata={"chunk_id": i, "type": "text", "source": filename}
            ))
        
        texts = [doc.page_content for doc in temp_docs]
        embeddings = await embedding_client.embed(texts=texts)
        temp_embeddings = embeddings if isinstance(embeddings[0], list) else [embeddings]
        
    else:
        raise ValueError(f"Unsupported file type: {filename}")
    
    # Store in Pinecone
    vectors = [
        {
            "id": f"doc-{i}",
            "values": emb if isinstance(emb, list) else emb.tolist(),
            "metadata": {
                "text": doc.page_content,  # Store text for retrieval
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": i
            }
        }
        for i, (doc, emb) in enumerate(zip(temp_docs, temp_embeddings))
    ]
    
    # Create namespace from filename
    name = os.path.splitext(filename)[0]
    current_namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]
    
    # Upsert to Pinecone (batched automatically)
    print("Please wait. . . . . .")
    await asyncio.to_thread(
        pc_index.upsert,
        vectors=vectors,
        namespace=current_namespace
    )
    
    elapsed = time.time() - start_time
    print(f"✅ Stored {len(temp_docs)} chunks in namespace '{current_namespace}' ({elapsed:.2f}s)")
    
    result_docs = temp_docs.copy()
    temp_docs.clear()
    temp_embeddings.clear()
    
    return result_docs

# =============================================================================
# OPTIMIZED PINECONE RETRIEVER
# =============================================================================
class FastPineconeRetriever:
    """Ultra-fast Pinecone retriever optimized for text-only RAG"""
    
    def __init__(self, index, embedding_client, top_k: int = 3, namespace: str = "__default__"):
        self.index = index
        self.embedding_client = embedding_client
        self.top_k = top_k
        self.default_namespace = namespace
    
    async def __call__(self, query: str, namespace: Optional[str] = None):
        """Async retrieval"""
        return await self.get_relevant_documents(query, namespace)
    
    async def get_relevant_documents(self, query: str, namespace: Optional[str] = None):
        """Retrieve documents with aggressive optimization"""
        from langchain_core.documents import Document
        
        if not isinstance(query, str) or not query.strip():
            return []
        
        use_namespace = namespace or current_namespace or self.default_namespace
        
        # Get embedding
        start = time.time()
        dense_vec = await self.embedding_client.embed(text=query.strip())
        embed_time = time.time() - start

        print(f"Quey done in {embed_time}")
        
        # Query Pinecone
        start = time.time()
        
        try:
            result = await asyncio.to_thread(
                self.index.query,
                vector=dense_vec,
                top_k=self.top_k,
                include_metadata=True,
                namespace=use_namespace
            )
            query_time = time.time() - start
            
            print(f"⚡ Retrieval: embed={embed_time:.0f}s, query={query_time:.0f}s, matches={len(result.matches)}")
            
            if not result.matches:
                print(f"⚠️ No matches in namespace '{use_namespace}'")
                return []
            
            # Convert to documents
            docs = [
                Document(
                    page_content=m.metadata.get("text", ""),
                    metadata={
                        "score": m.score,
                        "source": m.metadata.get("source", "unknown"),
                        "id": m.id
                    }
                )
                for m in result.matches
                if m.metadata.get("text")  # Only return docs with text
            ]
            
            return docs
            
        except Exception as e:
            print(f"❌ Pinecone error: {e}")
            return []

# =============================================================================
# LLM RESPONSE GENERATION - STREAMING
# =============================================================================
async def generate_response(question: str, session_id: str = "default"):
    """Generate response with full optimization"""
    
    # Get conversation context (async)
    context_start = time.time()
    conversation_history = await get_conversation_context(session_id, include_mongo=False)
    context_text = format_context_for_model(conversation_history)
    context_time = time.time() - context_start
    
    # Check cache
    cache_key = get_cache_key(question, context_text, current_namespace)
    if cache_key in query_cache:
        print(f"✅ Cache hit")
        yield query_cache[cache_key]
        return
    
    # Build input
    chain_input = {
        "question": question,
        "chat_history": context_text
    }
    
    try:
        start = time.time()
        
        full_response = ""
        async for chunk in final_chain.astream(chain_input):
            full_response += chunk
            yield chunk
        
        total_time = time.time() - start
        
        print(f"⚡ Total: context={context_time:.0f}s, generation={total_time:.0f}s")
        
        # Cache the result
        query_cache[cache_key] = full_response
        
    except Exception as e:
        print(f"❌ Generation error: {e}")
        traceback.print_exc()
        yield "I apologize, but I encountered an error. Please try again."

from pinecone import Pinecone
pinecone_storage = Pinecone(
        api_key=PINECONE_KEY,
        pool_threads=50,
        timeout=8
    )
pc_index = pinecone_storage.Index("rag-img-text-db")

# =============================================================================
# APPLICATION STARTUP
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all resources"""
    global llm, pc_index, pc_retriever, final_chain, embedding_client
    
    start_time = time.time()
    print("\n" + "="*70)
    print("🚀 RAG SYSTEM STARTUP")
    print("="*70)
    
    # 1. Initialize storage
    print("\n⏱️ Initializing storage...")
    init_storage()
    
    # 2. Initialize embedding client
    print("\n⏱️ Initializing embedding client...")
    embedding_client = EmbeddingClient(f"{EMBED_SERVICE_URL}/embed")
    
    # 3. Import libraries
    print("\n⏱️ Importing libraries...")
    from pinecone import Pinecone
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    
    # 4. Initialize LLM with optimizations
    print("\n⏱️ Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-exp",  # Latest fastest model
        google_api_key=GOOGLE_API_KEY,
        streaming=True,
        request_timeout=15
    )
    print("✅ LLM ready")
    
    # 5. Initialize Pinecone with optimizations
    print("\n⏱️ Setting up Pinecone...")
    pinecone_storage = Pinecone(
        api_key=PINECONE_KEY,
        pool_threads=50,
        timeout=8
    )
    
    pc_index = pinecone_storage.Index("rag-img-text-db")
    
    pc_retriever = FastPineconeRetriever(
        index=pc_index,
        embedding_client=embedding_client,
        top_k=TOP_K_RETRIEVAL,
        namespace="__default__"
    )
    print("✅ Pinecone ready")
    
    # 6. Build optimized RAG chain
    print("\n⏱️ Building RAG chain...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant. Use the provided context to answer questions accurately and concisely.

RULES:
- If context is relevant, use it to answer
- If context is insufficient, use your knowledge
- Be direct and concise
- Cite sources when using context"""),
        ("user", """Context: {rag_context}

Previous conversation: {chat_history}

Question: {question}

Answer:""")
    ])
    
    # Async retrieval wrapper
    async def async_retrieve(input_dict):
        question = input_dict.get("question", "")
        docs = await pc_retriever.get_relevant_documents(question)
        if not docs:
            return "No relevant context found."
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Build chain with async retrieval
    final_chain = (
        RunnableParallel({
            "rag_context": RunnableLambda(async_retrieve),
            "chat_history": lambda x: x.get("chat_history", ""),
            "question": lambda x: x.get("question", "")
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain ready")

    print("conecting to embed service. . . ")
    import requests
    testing = requests.get(url="http://embedding_service:8000/health")
    print(testing.json())
    
    elapsed = time.time() - start_time
    app.state.startup_time = elapsed
    
    print("\n" + "="*70)
    print(f"✅ SYSTEM READY! Startup: {elapsed:.2f}s")
    print("="*70 + "\n")
    
    yield
    
    # Cleanup
    print("\n🛑 Shutting down...")
    if embedding_client:
        await embedding_client.close()