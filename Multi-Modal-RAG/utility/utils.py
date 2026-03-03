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
EMBEDDING_BATCH_SIZE = 32
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
collection = None        # MongoDB conversations collection (optional fallback)
sessions_collection = None  # MongoDB sessions collection (optional fallback)
mongo_client_g = None
redis_client = None
embedding_client = None
pc_index = None
pc_retriever = None
llm = None
final_chain = None
current_namespace = "__default__"

# ── Worker pools ──────────────────────────────────────────────────────────────
_process_pool = ProcessPoolExecutor(max_workers=max(2, os.cpu_count() - 1))
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
    global collection, sessions_collection, mongo_client_g, redis_client
    try:
        from pymongo import MongoClient, ASCENDING, DESCENDING
        from pymongo.server_api import ServerApi
        import redis as _redis

        mongo_client_g = MongoClient(
            MONGO_URL, server_api=ServerApi("1"), maxPoolSize=20, minPoolSize=2
        )
        db = mongo_client_g["chat-collection"]

        # Conversations collection (messages)
        collection = db["conversations"]
        try:
            collection.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="session_ts_idx",
                background=True,
            )
        except Exception:
            pass

        # NEW: Sessions collection (metadata per session)
        sessions_collection = db["sessions"]
        try:
            sessions_collection.create_index(
                [("session_id", ASCENDING)],
                name="session_id_idx",
                unique=True,
                background=True,
            )
            sessions_collection.create_index(
                [("updated_at", DESCENDING)],
                name="updated_at_idx",
                background=True,
            )
        except Exception:
            pass

        print("✅ MongoDB connected (conversations + sessions)")

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
        sessions_collection = None
        redis_client = None


# =============================================================================
# SESSION MANAGEMENT  (NEW)
# =============================================================================

def _session_meta_key(session_id: str) -> str:
    return f"session:{session_id}:meta"


def _session_messages_key(session_id: str) -> str:
    return f"chat:{session_id}:messages"


SESSIONS_INDEX_KEY = "sessions:index"

def _create_session_sync(session_id: str, title: str = "New Chat") -> Dict:
    """Create a session in Redis if it doesn't already exist."""
    if redis_client is None:
        return {}
    try:
        now = time.time()
        meta_key = _session_meta_key(session_id)
        created = False

        if not redis_client.exists(meta_key):
            redis_client.hset(
                meta_key,
                mapping={
                    "session_id": session_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                    "preview": "",
                },
            )
            created = True

        updated_at = float(redis_client.hget(meta_key, "updated_at") or now)
        redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: updated_at})
        return {"session_id": session_id, "created": created}
    except Exception as e:
        print(f"⚠️ Session create failed: {e}")
        return {}


def _update_session_sync(session_id: str, user_message: str, assistant_preview: str):
    """Update Redis session metadata after a new exchange."""
    if redis_client is None:
        return
    try:
        now = time.time()
        meta_key = _session_meta_key(session_id)
        new_title = user_message[:45] + ("…" if len(user_message) > 45 else "")

        if not redis_client.exists(meta_key):
            redis_client.hset(
                meta_key,
                mapping={
                    "session_id": session_id,
                    "title": new_title,
                    "created_at": now,
                    "updated_at": now,
                    "message_count": 0,
                    "preview": "",
                },
            )

        current_title = redis_client.hget(meta_key, "title") or "New Chat"
        updates: Dict[str, Any] = {
            "updated_at": now,
            "preview": assistant_preview[:80],
        }
        if current_title == "New Chat":
            updates["title"] = new_title

        message_count = int(redis_client.hget(meta_key, "message_count") or 0) + 2
        updates["message_count"] = message_count

        redis_client.hset(meta_key, mapping=updates)
        redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: now})
    except Exception as e:
        print(f"⚠️ Session update failed: {e}")


def _get_all_sessions_sync(limit: int = 50) -> List[Dict]:
    """Fetch all sessions from Redis ordered by most recently updated."""
    if redis_client is None:
        return []
    try:
        session_ids = redis_client.zrevrange(SESSIONS_INDEX_KEY, 0, max(0, limit - 1))
        docs: List[Dict[str, Any]] = []
        for sid in session_ids:
            meta = redis_client.hgetall(_session_meta_key(sid))
            if not meta:
                continue
            docs.append(
                {
                    "session_id": sid,
                    "title": meta.get("title", "New Chat"),
                    "preview": meta.get("preview", ""),
                    "updated_at": float(meta.get("updated_at", 0) or 0),
                    "created_at": float(meta.get("created_at", 0) or 0),
                    "message_count": int(meta.get("message_count", 0) or 0),
                }
            )
        return docs
    except Exception as e:
        print(f"⚠️ Get sessions failed: {e}")
        return []


def _delete_session_sync(session_id: str):
    """Delete a session and all its messages from Redis (+ optional Mongo fallback)."""
    try:
        if redis_client is not None:
            redis_client.delete(_session_messages_key(session_id))
            redis_client.delete(_session_meta_key(session_id))
            redis_client.zrem(SESSIONS_INDEX_KEY, session_id)
        if sessions_collection is not None:
            sessions_collection.delete_one({"session_id": session_id})
        if collection is not None:
            collection.delete_many({"session_id": session_id})
    except Exception as e:
        print(f"⚠️ Delete session failed: {e}")


def _rename_session_sync(session_id: str, new_title: str):
    if redis_client is None and sessions_collection is None:
        return
    try:
        if redis_client is not None:
            meta_key = _session_meta_key(session_id)
            if redis_client.exists(meta_key):
                redis_client.hset(meta_key, "title", new_title[:60])
                updated_at = float(redis_client.hget(meta_key, "updated_at") or time.time())
                redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: updated_at})
        if sessions_collection is not None:
            sessions_collection.update_one(
                {"session_id": session_id},
                {"$set": {"title": new_title[:60]}},
            )
    except Exception as e:
        print(f"⚠️ Rename session failed: {e}")


async def get_all_sessions(limit: int = 50) -> List[Dict]:
    return await asyncio.to_thread(_get_all_sessions_sync, limit)


async def create_session(session_id: str, title: str = "New Chat") -> Dict:
    return await asyncio.to_thread(_create_session_sync, session_id, title)


async def delete_session(session_id: str):
    await asyncio.to_thread(_delete_session_sync, session_id)


async def rename_session(session_id: str, new_title: str):
    await asyncio.to_thread(_rename_session_sync, session_id, new_title)


# =============================================================================
# STORAGE HELPERS – Redis + MongoDB
# =============================================================================
@traceable(name="save msg to redis")
def save_message_redis(session_id: str, role: str, content: str):
    if redis_client is None:
        return
    try:
        key = _session_messages_key(session_id)
        msg = json.dumps({"role": role, "content": content, "timestamp": time.time()})
        redis_client.rpush(key, msg)
        # Keep enough history for per-session chat replay in UI.
        redis_client.ltrim(key, -2000, -1)
        redis_client.expire(key, 60 * 60 * 24 * 30)
    except Exception:
        pass


@traceable(name="save msg to mongodb")
def save_message_mongo(session_id: str, role: str, content: str):
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


@traceable(name="getting msg from redis")
def get_short_memory(session_id: str) -> list:
    if redis_client is None:
        return []
    try:
        key = _session_messages_key(session_id)
        return [json.loads(m) for m in redis_client.lrange(key, -MAX_SHORT_MEMORY, -1)]
    except Exception:
        return []


@traceable(name="getting msg from mongo")
def get_mongo_history(session_id: str, limit: int = 100) -> list:
    if collection is None:
        return []
    try:
        msgs = list(
            collection.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
            )
            .sort("timestamp", 1)  # ascending for full history display
            .limit(limit)
        )
        return msgs
    except Exception:
        return []


@traceable(name="getting full conversation")
async def get_conversation_context(session_id: str, include_mongo: bool = True) -> list:
    recent = await asyncio.to_thread(get_short_memory, session_id)
    return recent[-MAX_HISTORY_CONTEXT:]


async def get_full_session_history(session_id: str) -> list:
    """Fetch complete message history for a session from Redis (for UI display)."""
    if redis_client is None:
        return []
    try:
        msgs = await asyncio.to_thread(redis_client.lrange, _session_messages_key(session_id), 0, -1)
        parsed = [json.loads(m) for m in msgs]
        return [
            {"role": m.get("role", "assistant"), "content": m.get("content", ""), "timestamp": m.get("timestamp", 0)}
            for m in parsed
        ]
    except Exception:
        return []


@traceable(name="formatting conversation")
def format_context_for_model(messages: list) -> str:
    if not messages:
        return ""
    parts = [
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in messages[-4:]
    ]
    return "\n".join(parts)


# =============================================================================
# EMBEDDING CLIENT
# =============================================================================
class EmbeddingClient:
    def __init__(self, service_url: str):
        self.service_url = service_url.rstrip("/")
        self.session: Optional[aiohttp.ClientSession] = None
        self.timeout = aiohttp.ClientTimeout(total=120, connect=5)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            connector = aiohttp.TCPConnector(
                limit=200, limit_per_host=100,
                ttl_dns_cache=300, keepalive_timeout=60,
            )
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=self.timeout,
            )
        return self.session

    @traceable(name="call embedding")
    async def embed(self, text=None, texts=None) -> Any:
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
            return [[0.0] * dim for _ in texts] if texts else [0.0] * dim
        except Exception as e:
            dim = 768
            return [[0.0] * dim for _ in texts] if texts else [0.0] * dim

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()


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
    def __init__(self, index, emb_client: EmbeddingClient,
                 top_k: int = TOP_K_RETRIEVAL, namespace: str = "__default__"):
        self.index = index
        self.emb_client = emb_client
        self.top_k = top_k
        self.default_namespace = namespace

    async def __call__(self, query: str, namespace: Optional[str] = None):
        return await self.get_relevant_documents(query, namespace)

    async def get_relevant_documents(self, query: str, namespace: Optional[str] = None):
        from langchain_core.documents import Document
        if not query or not query.strip():
            return []
        ns = namespace or current_namespace or self.default_namespace
        query = query.strip()
        dense_task = asyncio.create_task(self._dense_search(query, ns))
        sparse_task = asyncio.create_task(self._sparse_search(query, ns))
        dense_hits, sparse_hits = await asyncio.gather(dense_task, sparse_task)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits)[: self.top_k]
        docs, seen = [], set()
        for doc_id, score, meta in fused:
            text = meta.get("text", "")
            if not text or text in seen:
                continue
            seen.add(text)
            docs.append(Document(
                page_content=text,
                metadata={"score": score, "source": meta.get("source", "unknown"), "id": doc_id},
            ))
        return docs

    async def _dense_search(self, query: str, namespace: str) -> List[Tuple[str, float, Dict]]:
        try:
            vec = await self.emb_client.embed(text=query)
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self.index.query, vector=vec, top_k=self.top_k * 2,
                    include_metadata=True, namespace=namespace,
                ),
                timeout=8.0,
            )
            return [(m.id, m.score, m.metadata) for m in result.matches if m.metadata.get("text")]
        except Exception as e:
            print(f"⚠️ Dense search error: {e}")
            return []

    async def _sparse_search(self, query: str, namespace: str) -> List[Tuple[str, float, Dict]]:
        try:
            bm25 = _get_bm25(namespace)
            if not bm25.docs:
                return []
            loop = asyncio.get_event_loop()
            hits = await loop.run_in_executor(_thread_pool, lambda: bm25.query(query, top_k=self.top_k * 2))
            return [
                (f"bm25-{idx}", score, {**bm25.metadata[idx], "text": bm25.docs[idx]})
                for idx, score in hits
            ]
        except Exception as e:
            print(f"⚠️ BM25 search error: {e}")
            return []


# =============================================================================
# DOCUMENT EXTRACTION
# =============================================================================
def _extract_pdf_worker(pdf_bytes: bytes, filename: str) -> List[Dict]:
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    full = "\n\n".join(p.get_text() for p in doc if p.get_text().strip())
    return [{"text": c, "source": filename, "chunk_id": i} for i, c in enumerate(splitter.split_text(full))]


def _extract_text_worker(text: str, filename: str) -> List[Dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [{"text": c, "source": filename, "chunk_id": i} for i, c in enumerate(splitter.split_text(text))]


# =============================================================================
# EMBEDDING BATCHES
# =============================================================================
async def _embed_chunks_parallel(chunks: List[Dict]) -> List[Dict]:
    texts = [c["text"] for c in chunks]
    batches = [texts[i: i + EMBEDDING_BATCH_SIZE] for i in range(0, len(texts), EMBEDDING_BATCH_SIZE)]
    t0 = time.time()
    results = await asyncio.gather(*[embedding_client.embed(texts=b) for b in batches], return_exceptions=True)
    print(f"  🔢 {len(texts)} chunks embedded in {time.time()-t0:.2f}s ({len(batches)} concurrent batches)")
    all_embeddings: List[List[float]] = []
    for r in results:
        if isinstance(r, Exception):
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
async def _upsert_to_pinecone(chunks: List[Dict], namespace: str):
    vectors = [
        {
            "id": f"{namespace}-{c['source']}-{c['chunk_id']}",
            "values": c["embedding"],
            "metadata": {"text": c["text"], "source": c["source"], "chunk_id": c["chunk_id"]},
        }
        for c in chunks
    ]
    sub_batches = [vectors[i: i + 100] for i in range(0, len(vectors), 100)]
    await asyncio.gather(
        *[asyncio.to_thread(pc_index.upsert, vectors=sb, namespace=namespace) for sb in sub_batches]
    )


# =============================================================================
# FILE UPLOAD
# =============================================================================
async def extract_and_store(filename: str, file_obj, target_namespace: Optional[str] = None) -> List[Dict]:
    global current_namespace
    t0 = time.time()
    loop = asyncio.get_event_loop()
    print(f"📄 Processing: {filename}")

    if filename.lower().endswith(".pdf"):
        pdf_bytes = file_obj.file.read()
        chunks = await loop.run_in_executor(_process_pool, _extract_pdf_worker, pdf_bytes, filename)
    elif filename.lower().endswith((".txt", ".md")):
        raw_text = file_obj.file.read().decode("utf-8")
        chunks = await loop.run_in_executor(_process_pool, _extract_text_worker, raw_text, filename)
    else:
        raise ValueError(f"Unsupported: {filename}")

    print(f"  ✂️  {len(chunks)} chunks extracted ({time.time()-t0:.2f}s)")
    chunks = await _embed_chunks_parallel(chunks)

    if target_namespace:
        namespace = target_namespace
    else:
        name = os.path.splitext(filename)[0]
        namespace = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:63]
        current_namespace = namespace

    await _upsert_to_pinecone(chunks, namespace)
    await loop.run_in_executor(
        _thread_pool, _update_bm25, namespace,
        [c["text"] for c in chunks],
        [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks],
    )
    print(f"✅ {len(chunks)} chunks → ns='{namespace}' ({time.time()-t0:.2f}s)")
    return chunks


async def extract_and_store_multiple(files: List[Tuple[str, Any]]) -> Dict[str, Any]:
    global current_namespace
    names = [re.sub(r"[^a-zA-Z0-9_-]", "_", os.path.splitext(fn)[0])[:20] for fn, _ in files]
    combined_ns = "__".join(names)[:63]
    current_namespace = combined_ns
    print(f"📦 Multi-upload → combined namespace: '{combined_ns}'")
    tasks = [asyncio.create_task(extract_and_store(fn, fo, target_namespace=combined_ns)) for fn, fo in files]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    summary: Dict[str, Any] = {}
    for (fn, _), result in zip(files, results):
        if isinstance(result, Exception):
            summary[fn] = {"status": "error", "error": str(result)}
        else:
            summary[fn] = {"status": "ok", "chunks": len(result)}
    return summary


# =============================================================================
# LLM RESPONSE GENERATION
# =============================================================================
async def generate_response(
    question: str,
    session_id: str = "default",
    image_base64: Optional[str] = None,
):
    """Streaming generator. Saves to Redis, MongoDB, and updates session metadata."""
    # Ensure session exists
    await asyncio.to_thread(_create_session_sync, session_id)

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

        # Persist messages
        save_message_redis(session_id, "user", question)
        save_message_redis(session_id, "assistant", full_response)
        # Mongo persistence kept optional; Redis is the source of truth for session replay.

        # Update session metadata
        await asyncio.to_thread(_update_session_sync, session_id, question, full_response[:80])

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

    init_storage()

    embedding_client = EmbeddingClient(EMBED_SERVICE_URL)
    import requests as _req
    try:
        h = _req.get(f"{EMBED_SERVICE_URL}/health", timeout=5)
        print(f"✅ Embedding service: {h.json()}")
    except Exception as e:
        print(f"⚠️ Embedding service unreachable: {e}")

    from pinecone.grpc import PineconeGRPC
    pc = PineconeGRPC(api_key=PINECONE_KEY, pool_threads=50, timeout=10)
    pc_index = pc.Index(RAG_INDEX)
    pc_retriever = HybridPineconeRetriever(
        index=pc_index, emb_client=embedding_client,
        top_k=TOP_K_RETRIEVAL, namespace="__default__",
    )
    print("✅ Pinecone + Hybrid retriever ready")

    from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
    chat_llm = HuggingFaceEndpoint(
        repo_id="deepseek-ai/DeepSeek-R1", huggingfacehub_api_token=HF_TOKEN
    )
    llm = ChatHuggingFace(llm=chat_llm)
    print("✅ LLM ready")

    from langchain_core.runnables import RunnableParallel, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant with access to document context.

RULES:
- Use context when relevant; otherwise don't answer this question.
- Be direct, accurate, and concise.
- Cite source filenames when referencing context.
- If an image is described, incorporate it into your answer."""),
        ("user", """Context:\n{rag_context}

Conversation history:\n{chat_history}

Question: {question}

Answer:"""),
    ])

    async def async_retrieve(inp: dict) -> str:
        docs = await pc_retriever.get_relevant_documents(inp.get("question", ""))
        if not docs:
            return "No relevant context found."
        return "\n\n".join(d.page_content for d in docs)

    final_chain = (
        RunnableParallel({
            "rag_context": RunnableLambda(async_retrieve),
            "chat_history": lambda x: x.get("chat_history", ""),
            "question": lambda x: x.get("question", ""),
        })
        | prompt
        | llm
        | StrOutputParser()
    )
    print("✅ RAG chain ready")

    elapsed = time.time() - t0
    app.state.startup_time = elapsed
    print(f"\n✅ SYSTEM READY in {elapsed:.2f}s\n" + "=" * 70)

    yield

    print("\n🛑 Shutting down…")
    if embedding_client:
        await embedding_client.close()
    _process_pool.shutdown(wait=False)
    _thread_pool.shutdown(wait=False)
