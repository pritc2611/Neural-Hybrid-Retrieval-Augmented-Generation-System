"""
config.py — Constants, environment variables, global state handles, worker pools, caches.
"""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from cachetools import TTLCache
from dotenv import load_dotenv

load_dotenv()

# Retrieval tuning
MAX_SHORT_MEMORY = 20
MAX_HISTORY_CONTEXT = 6
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100
TOP_K_RETRIEVAL = 10
EMBEDDING_BATCH_SIZE = 32
DENSE_WEIGHT = 0.6
SPARSE_WEIGHT = 0.4

# Environment
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
MONGO_URL = os.getenv("MONGO_URL")
RAG_INDEX = os.getenv("RAG_INDEX")
EMBED_SERVICE_URL = os.getenv("EMBED_SERVICE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
REDIS_HOST = os.getenv("REDIS_HOST")
HF_TOKEN = os.getenv("HF_TOKEN")
REDIS_PASS = os.getenv("REDIS_PASS")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# Global handles (populated at lifespan startup)
collection = None
sessions_collection = None
mongo_client_g = None
redis_client = None
embedding_client = None
pc_index = None
pc_retriever = None
llm = None
final_chain = None
current_namespace = "__default__"

# Worker pools
_process_pool = ProcessPoolExecutor(max_workers=max(2, os.cpu_count() - 1))
_thread_pool = ThreadPoolExecutor(max_workers=16)

# Caches
query_cache = TTLCache(maxsize=2000, ttl=1800)
embedding_cache = TTLCache(maxsize=5000, ttl=3600)
