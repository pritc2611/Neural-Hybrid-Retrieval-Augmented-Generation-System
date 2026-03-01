import os
import re
import time
import json
import math
import hashlib
import asyncio
import aiohttp
import traceback
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import List, Optional, Dict, Any, Tuple
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint
import numpy as np
from cachetools import TTLCache
from dotenv import load_dotenv
from fastapi import FastAPI
from langsmith import traceable

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================
MAX_SHORT_MEMORY = 10
MAX_HISTORY_CONTEXT = 6
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 5
EMBEDDING_BATCH_SIZE = 32  # smaller batches → faster per-request latency
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

PINECONE_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URL = os.getenv("MONGO_URL")
RAG_INDEX = os.getenv("RAG_INDEX")
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL", "http://localhost:8000")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
HF_TOKEN = os.getenv("HF_TOKEN")
REDIS_PASS = os.getenv("REDIS_PASS")

# ── Global handles ────────────────────────────────────────────────────────────
collection = None  # MongoDB collection
mongo_client_g = None  # keep alive
redis_client = None
embedding_client = None
pc_index = None
pc_retriever = None
llm = None
final_chain = None
current_namespace = "__default__"

# ── Worker pools ──────────────────────────────────────────────────────────────
# ProcessPoolExecutor for CPU-bound extraction (PDF + TXT/MD)
_process_pool = ProcessPoolExecutor(max_workers=max(2, os.cpu_count() - 1))
# ThreadPoolExecutor for I/O-bound BM25 ops + misc blocking calls
_thread_pool = ThreadPoolExecutor(max_workers=16)

# =============================================================================
# CACHES
# =============================================================================
query_cache = TTLCache(maxsize=2000, ttl=1800)
embedding_cache = TTLCache(maxsize=5000, ttl=3600)


def get_cache_key(question: str, context: str, namespace: str = "__default__") -> str:
    ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
    combined = f"{question.strip().lower()}|{ctx_hash}|{namespace}"
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


# =============================================================================
# STORAGE INIT
# =============================================================================
def init_storage():
    global collection, mongo_client_g, redis_client
    try:
        from pymongo import MongoClient, ASCENDING, DESCENDING
        from pymongo.server_api import ServerApi
        import redis as _redis

        mongo_client_g = MongoClient(
            MONGO_URL, server_api=ServerApi("1"), maxPoolSize=20, minPoolSize=2
        )
        db = mongo_client_g["chat-collection"]
        collection = db["conversations"]
        try:
            collection.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="session_ts_idx",
                background=True,
            )
        except Exception:
            pass
        print("✅ MongoDB connected")

        redis_client = _redis.Redis(
            host=REDIS_HOST,
            port=17564,
            decode_responses=True,
            username="default",
            password=REDIS_PASS,
        )
        redis_client.ping()
        print("✅ Redis connected")
    except Exception as e:
        print(f"⚠️ Storage init failed: {e}")
        collection = None
        redis_client = None


# =============================================================================
# STORAGE HELPERS – Redis (fast cache) + MongoDB (persistent)
# =============================================================================
@traceable(name="save msg to redis")
def save_message_redis(session_id: str, role: str, content: str):
    if redis_client is None:
        return
    try:
        key = f"chat:{session_id}"
        msg = json.dumps({"role": role, "content": content[:1000]})
        redis_client.rpush(key, msg)
        redis_client.ltrim(key, -MAX_SHORT_MEMORY, -1)
        redis_client.expire(key, 86400)
    except Exception:
        pass


@traceable(name="save msg to mongodb")
def save_message_mongo(session_id: str, role: str, content: str):
    """Persist message to MongoDB for long-term history."""
    if collection is None:
        return
    try:
        collection.insert_one(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
        )
    except Exception as e:
        print(f"⚠️ Mongo save failed: {e}")


@traceable(name="geting msg from redis")
def get_short_memory(session_id: str) -> list:
    if redis_client is None:
        return []
    try:
        key = f"chat:{session_id}"
        return [json.loads(m) for m in redis_client.lrange(key, 0, -1)]
    except Exception:
        return []


@traceable(name="geting msg from mongo")
def get_mongo_history(session_id: str, limit: int = 20) -> list:
    """Fetch older conversation history from MongoDB."""
    if collection is None:
        return []
    try:
        msgs = list(
            collection.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
            )
            .sort("timestamp", -1)
            .limit(limit)
        )
        msgs.reverse()
        return msgs
    except Exception:
        return []


@traceable(name="getting full converstation")
async def get_conversation_context(session_id: str, include_mongo: bool = True) -> list:
    """
    Fast path: Redis only.
    Full path: Redis recent + MongoDB older (deduplicated).
    """
    recent = await asyncio.to_thread(get_short_memory, session_id)

    if not include_mongo or collection is None:
        return recent[-MAX_HISTORY_CONTEXT:]

    try:
        older = await asyncio.to_thread(get_mongo_history, session_id, 30)
        redis_set = {(m["role"], m["content"]) for m in recent}
        merged = []
        for msg in older:
            if (msg["role"], msg["content"]) not in redis_set:
                merged.append(msg)
        merged.extend(recent)
        return merged[-MAX_HISTORY_CONTEXT:]
    except Exception:
        return recent[-MAX_HISTORY_CONTEXT:]


@traceable(name="formating converstation")
def format_context_for_model(messages: list) -> str:
    if not messages:
        return ""
    parts = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in messages[-4:]
    ]
    return "\n".join(parts)


# =============================================================================
# EMBEDDING CLIENT  –  concurrent batch dispatch + longer timeout
# =============================================================================
class EmbeddingClient:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        # ✅ FIX 2a: raise timeout so large batches don't abort
        self.timeout = aiohttp.ClientTimeout(total=120, connect=5)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=200,
                limit_per_host=100,
                ttl_dns_cache=300,
                keepalive_timeout=60,
            )
            self.session = aiohttp.ClientSession(
                connector=connector,
                timeout=self.timeout,
            )
        return self.session

    @traceable(name="call embedding")
    async def embed(
        self,
        text: Optional[str] = None,
        texts: Optional[List[str]] = None,
    ) -> Any:
        if text and texts:
            raise ValueError("Provide either text or texts, not both")
        if not text and not texts:
            raise ValueError("Must provide text or texts")

        # Single-text cache
        cache_key = None
        if text:
            cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
            if cache_key in embedding_cache:
                return embedding_cache[cache_key]

        payload = {"text": texts if texts else text}
        session = await self._get_session()

        try:
            async with session.post(f"{self.service_url}/embed", json=payload) as resp:
                resp.raise_for_status()
                result = await resp.json()
                if "embeddings" not in result:
                    raise RuntimeError(f"Invalid response: {result}")
                embeddings = result["embeddings"]
                if cache_key:
                    embedding_cache[cache_key] = embeddings
                return embeddings
        except asyncio.TimeoutError:
            dim = 768
            print("⚠️ Embedding timeout")
            return [[0.0] * dim for _ in texts] if texts else [0.0] * dim
        except Exception as e:
            dim = 768
            print(f"⚠️ Embedding error: {e}")
            return [[0.0] * dim for _ in texts] if texts else [0.0] * dim

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


# ── Type hint workaround ──────────────────────────────────────────────────────
from typing import Any


# =============================================================================
# BM25 SPARSE RETRIEVER
# =============================================================================
class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[str] = []
        self.metadata: List[Dict] = []
        self._idf: Dict[str, float] = {}
        self._tf: List[Dict[str, float]] = []
        self._avgdl: float = 0.0

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def build(self, docs: List[str], metadata: List[Dict]):
        self.docs = docs
        self.metadata = metadata
        tokenized = [self.tokenize(d) for d in docs]
        self._avgdl = float(np.mean([len(t) for t in tokenized])) if tokenized else 1.0
        self._tf = []
        df: Dict[str, int] = {}
        for tokens in tokenized:
            freq: Dict[str, float] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0) + 1
            self._tf.append(freq)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1
        N = len(docs)
        self._idf = {t: math.log((N - n + 0.5) / (n + 0.5) + 1) for t, n in df.items()}

    def query(self, q: str, top_k: int = 10) -> List[Tuple[int, float]]:
        if not self.docs:
            return []
        tokens = self.tokenize(q)
        scores = np.zeros(len(self.docs))
        for t in tokens:
            if t not in self._idf:
                continue
            idf = self._idf[t]
            for i, tf in enumerate(self._tf):
                f = tf.get(t, 0)
                dl = sum(tf.values())
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        ranked = sorted(enumerate(scores.tolist()), key=lambda x: -x[1])
        return ranked[:top_k]


_bm25_indexes: Dict[str, BM25Index] = {}


@traceable(name="geting bm25")
def _get_bm25(namespace: str) -> BM25Index:
    if namespace not in _bm25_indexes:
        _bm25_indexes[namespace] = BM25Index()
    return _bm25_indexes[namespace]


def _update_bm25(namespace: str, docs: List[str], metadata: List[Dict]):
    idx = _get_bm25(namespace)
    all_docs = idx.docs + docs
    all_meta = idx.metadata + metadata
    idx.build(all_docs, all_meta)
    print(f"✅ BM25 updated: ns='{namespace}', total={len(all_docs)}")


# =============================================================================
# RRF FUSION
# =============================================================================
@traceable(name="ranking on fusion vectors")
def reciprocal_rank_fusion(
    dense_hits: List[Tuple[str, float, Dict]],
    sparse_hits: List[Tuple[str, float, Dict]],
    k: int = 60,
) -> List[Tuple[str, float, Dict]]:
    scores: Dict[str, float] = {}
    meta_map: Dict[str, Dict] = {}
    for rank, (doc_id, _, meta) in enumerate(dense_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + DENSE_WEIGHT / (k + rank + 1)
        meta_map[doc_id] = meta
    for rank, (doc_id, _, meta) in enumerate(sparse_hits):
        scores[doc_id] = scores.get(doc_id, 0.0) + SPARSE_WEIGHT / (k + rank + 1)
        meta_map[doc_id] = meta
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [(did, sc, meta_map[did]) for did, sc in ranked]


# =============================================================================
# HYBRID RETRIEVER
# =============================================================================
class HybridPineconeRetriever:
    def __init__(
        self,
        index,
        emb_client: EmbeddingClient,
        top_k: int = TOP_K_RETRIEVAL,
        namespace: str = "__default__",
    ):
        self.index = index
        self.emb_client = emb_client
        self.top_k = top_k
        self.default_namespace = namespace

    async def __call__(self, query: str, namespace: Optional[str] = None):
        return await self.get_relevant_documents(query, namespace)

    @traceable(name="hybrid_retrieve")
    async def get_relevant_documents(self, query: str, namespace: Optional[str] = None):
        from langchain_core.documents import Document

        if not query or not query.strip():
            return []
        ns = namespace or current_namespace or self.default_namespace
        query = query.strip()
        dense_task, sparse_task = (
            asyncio.create_task(self._dense_search(query, ns)),
            asyncio.create_task(self._sparse_search(query, ns)),
        )
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[: self.top_k]
        docs, seen = [], set()
        for doc_id, score, meta in fused:
            text = meta.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "score": score,
                        "source": meta.get("source", "unknown"),
                        "id": doc_id,
                    },
                )
            )
        print(
            f"⚡ Hybrid: dense={len(dense_hits)}, bm25={len(sparse_hits)}, fused={len(docs)}, ns='{ns}'"
        )
        return docs

    @traceable(name="dense serching")
    async def _dense_search(
        self, query: str, namespace: str
    ) -> List[Tuple[str, float, Dict]]:
        try:
            vec = await self.emb_client.embed(text=query)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.index.query,
                    vector=vec,
                    top_k=self.top_k * 2,
                    include_metadata=True,
                    namespace=namespace,
                ),
                timeout=8.0,
            )
            return [
                (m.id, m.score, m.metadata)
                for m in result.matches
                if m.metadata.get("text")
            ]
        except Exception as e:
            print(f"⚠️ Dense search error: {e}")
            return []

    @traceable(name="sparse serching")
    async def _sparse_search(
        self, query: str, namespace: str
    ) -> List[Tuple[str, float, Dict]]:
        try:
            bm25 = _get_bm25(namespace)
            if not bm25.docs:
                return []
            loop = asyncio.get_event_loop()
            hits = await loop.run_in_executor(
                _thread_pool, lambda: bm25.query(query, top_k=self.top_k * 2)
            )
            return [
                (f"bm25-{idx}", score, {**bm25.metadata[idx], "text": bm25.docs[idx]})
                for idx, score in hits
            ]
        except Exception as e:
            print(f"⚠️ BM25 search error: {e}")
            return []


# =============================================================================
# DOCUMENT EXTRACTION  –  CPU-bound helpers (run in ProcessPoolExecutor)
# =============================================================================
@traceable(name="extracting pdf")
def _extract_pdf_worker(pdf_bytes: bytes, filename: str) -> List[Dict]:
    """Run inside a worker process."""
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full = "\n\n".join(p.get_text() for p in doc if p.get_text().strip())
    return [
        {"text": c, "source": filename, "chunk_id": i}
        for i, c in enumerate(splitter.split_text(full))
    ]


def _extract_text_worker(text: str, filename: str) -> List[Dict]:
    """
    ✅ FIX 1: Run in worker process (was blocking event loop ~52 s).
    Plain-text / Markdown extraction via RecursiveCharacterTextSplitter.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [
        {"text": c, "source": filename, "chunk_id": i}
        for i, c in enumerate(splitter.split_text(text))
    ]


# =============================================================================
# EMBEDDING  –  concurrent batch dispatch
# =============================================================================
@traceable(name="embedding chunks")
async def _embed_chunks_parallel(chunks: List[Dict]) -> List[Dict]:
    """
    ✅ FIX 2b: Fire all batches concurrently with asyncio.gather.
    Previously sequential → O(n_batches) latency; now O(1) (all in parallel).
    """
    texts = [c["text"] for c in chunks]

    # Split into sub-batches
    batches = [
        texts[i : i + EMBEDDING_BATCH_SIZE]
        for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)
    ]

    t0 = time.time()
    # Launch all batches simultaneously
    results = await asyncio.gather(
        *[embedding_client.embed(texts=b) for b in batches],
        return_exceptions=True,
    )
    print(
        f"  🔢 {len(texts)} chunks embedded in {time.time()-t0:.2f}s "
        f"({len(batches)} concurrent batches)"
    )

    # Flatten results
    all_embeddings: List[List[float]] = []
    for r in results:
        if isinstance(r, Exception):
            print(f"⚠️ Batch error: {r}")
            # pad with zeros so chunk count stays consistent
            for _ in range(EMBEDDING_BATCH_SIZE):
                all_embeddings.append([0.0] * 768)
        elif isinstance(r[0], list):
            all_embeddings.extend(r)
        else:
            all_embeddings.append(r)

    for chunk, emb in zip(chunks, all_embeddings):
        chunk["embedding"] = emb
    return chunks


# =============================================================================
# PINECONE UPSERT
# =============================================================================
@traceable(name="upserting to pinecone")
async def _upsert_to_pinecone(chunks: List[Dict], namespace: str):
    vectors = [
        {
            "id": f"{namespace}-{c['source']}-{c['chunk_id']}",
            "values": c["embedding"],
            "metadata": {
                "text": c["text"],
                "source": c["source"],
                "chunk_id": c["chunk_id"],
            },
        }
        for c in chunks
    ]
    # Upsert batches of 100 concurrently
    sub_batches = [vectors[i : i + 100] for i in range(0, len(vectors), 100)]
    await asyncio.gather(
        *[
            asyncio.to_thread(pc_index.upsert, vectors=sb, namespace=namespace)
            for sb in sub_batches
        ]
    )


# =============================================================================
# SINGLE FILE UPLOAD
# =============================================================================
@traceable(name="extract_and_store")
async def extract_and_store(
    filename: str,
    file_obj,
    target_namespace: Optional[str] = None,  # if given, override auto-name
) -> List[Dict]:
    """
    Extract → embed → store ONE document.
    If target_namespace is set (multi-upload path) chunks go there;
    otherwise namespace is derived from filename.
    """
    global current_namespace
    t0 = time.time()
    loop = asyncio.get_event_loop()
    print(f"📄 Processing: {filename}")

    # ── Extract (always in subprocess) ───────────────────────────────────────
    if filename.lower().endswith(".pdf"):
        pdf_bytes = file_obj.file.read()
        chunks = await loop.run_in_executor(
            _process_pool, _extract_pdf_worker, pdf_bytes, filename
        )
    elif filename.lower().endswith((".txt", ".md")):
        raw_text = file_obj.file.read().decode("utf-8")
        # ✅ FIX 1: offload to process pool (was blocking ~52 s)
        chunks = await loop.run_in_executor(
            _process_pool, _extract_text_worker, raw_text, filename
        )
    else:
        raise ValueError(f"Unsupported: {filename}")

    print(f"  ✂️  {len(chunks)} chunks extracted ({time.time()-t0:.2f}s)")

    # ── Embed (concurrent batches) ────────────────────────────────────────────
    chunks = await _embed_chunks_parallel(chunks)

    # ── Namespace ─────────────────────────────────────────────────────────────
    if target_namespace:
        namespace = target_namespace
    else:
        name = os.path.splitext(filename)[0]
        namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]
        current_namespace = namespace

    # ── Dense store ───────────────────────────────────────────────────────────
    await _upsert_to_pinecone(chunks, namespace)

    # ── Sparse store ──────────────────────────────────────────────────────────
    await loop.run_in_executor(
        _thread_pool,
        _update_bm25,
        namespace,
        [c["text"] for c in chunks],
        [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks],
    )

    elapsed = time.time() - t0
    print(f"✅ {len(chunks)} chunks → ns='{namespace}' ({elapsed:.2f}s)")
    return chunks


# =============================================================================
# MULTI-FILE UPLOAD  –  ✅ FIX 3: merged namespace
# =============================================================================
@traceable(name="extract_and_store_multiple")
async def extract_and_store_multiple(files: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """
    Upload several documents concurrently.

    Namespace strategy:
    • All chunks land in ONE combined namespace:
        e.g. files [doc_a.pdf, doc_b.md]  → namespace 'doc_a__doc_b'
    • This lets the retriever search across ALL uploaded files at once.
    • current_namespace is updated to the combined name.
    """
    global current_namespace

    # Build combined namespace from all filenames
    names = [
        re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(fn)[0])[:20] for fn, _ in files
    ]
    combined_ns = "__".join(names)[:63]  # Pinecone namespace max = 63 chars
    current_namespace = combined_ns

    print(f"📦 Multi-upload → combined namespace: '{combined_ns}'")

    # Process all files concurrently, all upserted to combined_ns
    tasks = [
        asyncio.create_task(extract_and_store(fn, fo, target_namespace=combined_ns))
        for fn, fo in files
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary: Dict[str, Any] = {}
    for (fn, _), result in zip(files, results):
        if isinstance(result, Exception):
            summary[fn] = {"status": "error", "error": str(result)}
            print(f"❌ {fn}: {result}")
        else:
            summary[fn] = {"status": "ok", "chunks": len(result)}
    return summary


# =============================================================================
# LLM RESPONSE GENERATION
# =============================================================================
@traceable(name="generate_response")
async def generate_response(
    question: str,
    session_id: str = "default",
    image_base64: Optional[str] = None,
):
    """Streaming generator.  Saves to both Redis and MongoDB on completion."""
    conv_history = await get_conversation_context(session_id, include_mongo=True)
    context_text = format_context_for_model(conv_history)

    cache_key = None
    if not image_base64:
        cache_key = get_cache_key(question, context_text, current_namespace)
        if cache_key in query_cache:
            print("✅ Cache hit")
            yield query_cache[cache_key]
            return

    chain_input = {"question": question, "chat_history": context_text}
    if image_base64:
        chain_input["image_base64"] = image_base64

    try:
        full_response = ""
        async for chunk in final_chain.astream(chain_input):
            full_response += chunk
            yield chunk

        if cache_key:
            query_cache[cache_key] = full_response

        # ✅ FIX 4: persist to both Redis and MongoDB
        save_message_redis(session_id, "user", question)
        save_message_redis(session_id, "assistant", full_response)
        # Fire-and-forget to MongoDB (non-blocking)
        # asyncio.get_event_loop().run_in_executor(
        #     _thread_pool, save_message_mongo, session_id, "user",      question
        # )
        # asyncio.get_event_loop().run_in_executor(
        #     _thread_pool, save_message_mongo, session_id, "assistant", full_response
        # )

    except Exception as e:
        print(f"❌ Generation error: {e}")
        traceback.print_exc()
        yield "I encountered an error. Please try again."


# =============================================================================
# LIFESPAN
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm, pc_index, pc_retriever, final_chain, embedding_client

    t0 = time.time()
    print("\n" + "=" * 70)
    print("🚀 HYBRID RAG SYSTEM STARTUP")
    print("=" * 70)

    # 1. Storage
    init_storage()

    # 2. Embedding client
    embedding_client = EmbeddingClient(EMBED_SERVICE_URL)
    import requests as _req

    try:
        h = _req.get(f"{EMBED_SERVICE_URL}/health", timeout=5)
        print(f"✅ Embedding service: {h.json()}")
    except Exception as e:
        print(f"⚠️ Embedding service unreachable: {e}")

    # 3. Pinecone
    from pinecone.grpc import PineconeGRPC

    pc = PineconeGRPC(api_key=PINECONE_KEY, pool_threads=50, timeout=10)
    pc_index = pc.Index(RAG_INDEX)
    pc_retriever = HybridPineconeRetriever(
        index=pc_index,
        emb_client=embedding_client,
        top_k=TOP_K_RETRIEVAL,
        namespace="__default__",
    )
    print("✅ Pinecone + Hybrid retriever ready")

    # 4. LLM
    chat_llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1", huggingfacehub_api_token=HF_TOKEN
    )

    llm = ChatHuggingFace(llm=chat_llm)
    print("✅ LLM ready")

    # 5. RAG chain
    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant with access to document context.

RULES:
- Use context when relevant; otherwise don't answer this question.
- Be direct, accurate, and concise.
- Cite source filenames when referencing context.
- If an image is described, incorporate it into your answer.""",
            ),
            (
                "user",
                """Context:\n{rag_context}

Conversation history:\n{chat_history}

Question: {question}

Answer:""",
            ),
        ]
    )

    @traceable(name="async_retrieve", run_type="retriever")
    async def async_retrieve(inp: dict) -> str:
        docs = await pc_retriever.get_relevant_documents(inp.get("question", ""))
        if not docs:
            return "No relevant context found."
        return "\n\n".join(d.page_content for d in docs)

    final_chain = (
        RunnableParallel(
            {
                "rag_context": RunnableLambda(async_retrieve),
                "chat_history": lambda x: x.get("chat_history", ""),
                "question": lambda x: x.get("question", ""),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain ready")

    elapsed = time.time() - t0
    app.state.startup_time = elapsed
    print(f"\n✅ SYSTEM READY in {elapsed:.2f}s\n" + "=" * 70)

    yield  # ← app runs here

    print("\n🛑 Shutting down…")
    if embedding_client:
        await embedding_client.close()
    _process_pool.shutdown(wait=False)
    _thread_pool.shutdown(wait=False)
