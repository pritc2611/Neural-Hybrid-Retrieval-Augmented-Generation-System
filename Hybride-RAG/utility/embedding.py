"""
embedding.py — Async HTTP client for the embedding microservice.
"""
import hashlib
import asyncio
import time
from typing import Any, List, Optional, Union
import aiohttp
from langsmith import traceable

def _get_cfg():
    import utility.config as cfg
    return cfg


def get_cache_key(question: str, context: str, namespace: str = "__default__") -> str:
    ctx_hash = hashlib.md5(context.encode()).hexdigest()[:8]
    combined = f"{question.strip().lower()}|{ctx_hash}|{namespace}"
    return hashlib.sha256(combined.encode()).hexdigest()[:24]


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
            self.session = aiohttp.ClientSession(connector=connector, timeout=self.timeout)
        return self.session

    @traceable(name="embed_http_request")
    async def embed(self, text=None, texts=None) -> Any:
        if text is not None and texts is not None:
            raise ValueError("Provide either text or texts, not both")
        if text is None and texts is None:
            raise ValueError("Must provide text or texts")

        cfg = _get_cfg()

        # --- cache check ---
        cache_key = None
        if text is not None:
            t0 = time.perf_counter()
            cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
            if cache_key in cfg.embedding_cache:
                print(f"⏱  [embed cache HIT]           {(time.perf_counter()-t0)*1000:.1f} ms")
                return cfg.embedding_cache[cache_key]
            print(f"⏱  [embed cache MISS build key] {(time.perf_counter()-t0)*1000:.1f} ms")

        # --- HTTP to embedding service (with retry) ---
        payload = {"text": texts if texts is not None else text}
        t0 = time.perf_counter()
        session = await self._get_session()
        t1 = time.perf_counter()
        print(f"⏱  [embed get_session]          {(t1-t0)*1000:.1f} ms")

        max_retries = 3
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                t_http = time.perf_counter()
                async with session.post(f"{self.service_url}/embed", json=payload) as resp:
                    resp.raise_for_status()
                    result = await resp.json()
                    print(f"⏱  [embed HTTP POST + parse]   {(time.perf_counter()-t_http)*1000:.1f} ms")

                if "embeddings" not in result:
                    raise RuntimeError(f"Invalid response from embedding service: {result}")
                embeddings = result["embeddings"]

                if cache_key:
                    cfg.embedding_cache[cache_key] = embeddings
                return embeddings

            except Exception as e:
                last_error = e
                print(f"   ⚠️ Embedding HTTP error (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(0.5 * attempt)

        # All retries exhausted — return zero vectors as fallback
        print(f"   ❌ Embedding service unreachable after {max_retries} retries: {last_error}")
        dim = 768
        return [[0.0] * dim for _ in texts] if texts is not None else [0.0] * dim

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()