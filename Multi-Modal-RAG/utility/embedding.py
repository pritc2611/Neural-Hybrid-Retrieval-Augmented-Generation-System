"""
embedding.py — Async HTTP client for the embedding microservice.
"""
import hashlib
import asyncio
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
    
    @traceable(name="creating embeddings")
    async def embed(self, text=None, texts=None) -> Any:
        if text and texts:
            raise ValueError("Provide either text or texts, not both")
        if not text and not texts:
            raise ValueError("Must provide text or texts")

        cfg = _get_cfg()
        cache_key = None
        if text:
            cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
            if cache_key in cfg.embedding_cache:
                return cfg.embedding_cache[cache_key]

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
                    cfg.embedding_cache[cache_key] = embeddings
                return embeddings
        except Exception:
            dim = 768
            return [[0.0] * dim for _ in texts] if texts else [0.0] * dim

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()