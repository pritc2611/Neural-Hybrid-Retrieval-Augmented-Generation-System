"""
storage.py — Redis + MongoDB init, session CRUD, message persistence.
"""
import time
import json
import asyncio
from typing import List, Dict, Any

from langsmith import traceable


def _get_cfg():
    import utility.config as cfg
    return cfg


# =============================================================================
# INIT
# =============================================================================
def init_storage():
    cfg = _get_cfg()
    try:
        from pymongo import MongoClient, ASCENDING, DESCENDING
        from pymongo.server_api import ServerApi
        import redis as _redis

        cfg.mongo_client_g = MongoClient(
            cfg.MONGO_URL, server_api=ServerApi("1"),
            maxPoolSize=20, minPoolSize=2,
        )
        db = cfg.mongo_client_g["chat-collection"]

        cfg.collection = db["conversations"]
        try:
            cfg.collection.create_index(
                [("session_id", ASCENDING), ("timestamp", DESCENDING)],
                name="session_ts_idx", background=True,
            )
        except Exception:
            pass

        cfg.sessions_collection = db["sessions"]
        try:
            cfg.sessions_collection.create_index(
                [("session_id", ASCENDING)],
                name="session_id_idx", unique=True, background=True,
            )
            cfg.sessions_collection.create_index(
                [("updated_at", DESCENDING)],
                name="updated_at_idx", background=True,
            )
        except Exception:
            pass

        print("✅ MongoDB connected (conversations + sessions)")

        cfg.redis_client = _redis.Redis(
            host=cfg.REDIS_HOST, port=17564,
            decode_responses=True, username="default", password=cfg.REDIS_PASS,
        )
        cfg.redis_client.ping()
        print("✅ Redis connected")

    except Exception as e:
        print(f"⚠️  Storage init failed: {e}")
        cfg.collection = None
        cfg.sessions_collection = None
        cfg.redis_client = None


# =============================================================================
# KEY HELPERS
# =============================================================================
SESSIONS_INDEX_KEY = "sessions:index"

def _session_meta_key(session_id: str)     -> str: return f"session:{session_id}:meta"
def _session_messages_key(session_id: str) -> str: return f"chat:{session_id}:messages"


# =============================================================================
# SESSION CRUD
# =============================================================================
def _create_session_sync(session_id: str, title: str = "New Chat") -> Dict:
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return {}
    try:
        now = time.time()
        meta_key = _session_meta_key(session_id)
        created = False
        if not cfg.redis_client.exists(meta_key):
            cfg.redis_client.hset(meta_key, mapping={
                "session_id": session_id, "title": title,
                "created_at": now, "updated_at": now,
                "message_count": 0, "preview": "",
            })
            created = True
        updated_at = float(cfg.redis_client.hget(meta_key, "updated_at") or now)
        cfg.redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: updated_at})
        return {"session_id": session_id, "created": created}
    except Exception as e:
        print(f"⚠️  Session create failed: {e}")
        return {}


def _update_session_sync(session_id: str, user_message: str, assistant_preview: str):
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return
    try:
        now = time.time()
        meta_key  = _session_meta_key(session_id)
        new_title = user_message[:45] + ("…" if len(user_message) > 45 else "")
        if not cfg.redis_client.exists(meta_key):
            cfg.redis_client.hset(meta_key, mapping={
                "session_id": session_id, "title": new_title,
                "created_at": now, "updated_at": now,
                "message_count": 0, "preview": "",
            })
        current_title = cfg.redis_client.hget(meta_key, "title") or "New Chat"
        updates: Dict[str, Any] = {"updated_at": now, "preview": assistant_preview[:80]}
        if current_title == "New Chat":
            updates["title"] = new_title
        updates["message_count"] = int(cfg.redis_client.hget(meta_key, "message_count") or 0) + 2
        cfg.redis_client.hset(meta_key, mapping=updates)
        cfg.redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: now})
    except Exception as e:
        print(f"⚠️  Session update failed: {e}")


def _get_all_sessions_sync(limit: int = 50) -> List[Dict]:
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return []
    try:
        session_ids = cfg.redis_client.zrevrange(SESSIONS_INDEX_KEY, 0, max(0, limit - 1))
        docs = []
        for sid in session_ids:
            meta = cfg.redis_client.hgetall(_session_meta_key(sid))
            if not meta:
                continue
            docs.append({
                "session_id":    sid,
                "title":         meta.get("title", "New Chat"),
                "preview":       meta.get("preview", ""),
                "updated_at":    float(meta.get("updated_at", 0) or 0),
                "created_at":    float(meta.get("created_at", 0) or 0),
                "message_count": int(meta.get("message_count", 0) or 0),
            })
        return docs
    except Exception as e:
        print(f"⚠️  Get sessions failed: {e}")
        return []


def _delete_session_sync(session_id: str):
    cfg = _get_cfg()
    try:
        if cfg.redis_client is not None:
            cfg.redis_client.delete(_session_messages_key(session_id))
            cfg.redis_client.delete(_session_meta_key(session_id))
            cfg.redis_client.zrem(SESSIONS_INDEX_KEY, session_id)
        if cfg.sessions_collection is not None:
            cfg.sessions_collection.delete_one({"session_id": session_id})
        if cfg.collection is not None:
            cfg.collection.delete_many({"session_id": session_id})
    except Exception as e:
        print(f"⚠️  Delete session failed: {e}")


def _rename_session_sync(session_id: str, new_title: str):
    cfg = _get_cfg()
    try:
        if cfg.redis_client is not None:
            meta_key = _session_meta_key(session_id)
            if cfg.redis_client.exists(meta_key):
                cfg.redis_client.hset(meta_key, "title", new_title[:60])
                updated_at = float(cfg.redis_client.hget(meta_key, "updated_at") or time.time())
                cfg.redis_client.zadd(SESSIONS_INDEX_KEY, {session_id: updated_at})
        if cfg.sessions_collection is not None:
            cfg.sessions_collection.update_one(
                {"session_id": session_id}, {"$set": {"title": new_title[:60]}},
            )
    except Exception as e:
        print(f"⚠️  Rename session failed: {e}")


# =============================================================================
# ASYNC SESSION API
# =============================================================================
async def get_all_sessions(limit: int = 50)                    -> List[Dict]: return await asyncio.to_thread(_get_all_sessions_sync, limit)
async def create_session(session_id: str, title: str = "New Chat") -> Dict:  return await asyncio.to_thread(_create_session_sync, session_id, title)
async def delete_session(session_id: str):                             await asyncio.to_thread(_delete_session_sync, session_id)
async def rename_session(session_id: str, new_title: str):             await asyncio.to_thread(_rename_session_sync, session_id, new_title)


# =============================================================================
# MESSAGE PERSISTENCE
# =============================================================================
@traceable(name="save msg to redis")
def save_message_redis(session_id: str, role: str, content: str):
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return
    try:
        key = _session_messages_key(session_id)
        msg = json.dumps({"role": role, "content": content, "timestamp": time.time()})
        cfg.redis_client.rpush(key, msg)
        cfg.redis_client.ltrim(key, -2000, -1)
        cfg.redis_client.expire(key, 60 * 60 * 24 * 30)
    except Exception:
        pass


@traceable(name="save msg to mongodb")
def save_message_mongo(session_id: str, role: str, content: str):
    cfg = _get_cfg()
    if cfg.collection is None:
        return
    try:
        cfg.collection.insert_one({
            "session_id": session_id, "role": role,
            "content": content, "timestamp": time.time(),
        })
    except Exception as e:
        print(f"⚠️  Mongo save failed: {e}")


# =============================================================================
# MESSAGE RETRIEVAL
# =============================================================================
@traceable(name="getting msg from redis")
def get_short_memory(session_id: str) -> list:
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return []
    try:
        key = _session_messages_key(session_id)
        return [json.loads(m) for m in cfg.redis_client.lrange(key, -cfg.MAX_SHORT_MEMORY, -1)]
    except Exception:
        return []


@traceable(name="getting msg from mongo")
def get_mongo_history(session_id: str, limit: int = 100) -> list:
    cfg = _get_cfg()
    if cfg.collection is None:
        return []
    try:
        return list(
            cfg.collection.find(
                {"session_id": session_id},
                {"_id": 0, "role": 1, "content": 1, "timestamp": 1},
            ).sort("timestamp", 1).limit(limit)
        )
    except Exception:
        return []


@traceable(name="getting full conversation")
async def get_conversation_context(session_id: str, include_mongo: bool = True) -> list:
    recent = await asyncio.to_thread(get_short_memory, session_id)
    return recent[-_get_cfg().MAX_HISTORY_CONTEXT:]


async def get_full_session_history(session_id: str) -> list:
    cfg = _get_cfg()
    if cfg.redis_client is None:
        return []
    try:
        msgs = await asyncio.to_thread(
            cfg.redis_client.lrange, _session_messages_key(session_id), 0, -1,
        )
        return [
            {"role": m.get("role", "assistant"), "content": m.get("content", ""), "timestamp": m.get("timestamp", 0)}
            for m in [json.loads(x) for x in msgs]
        ]
    except Exception:
        return []


def format_context_for_model(messages: list) -> str:
    if not messages:
        return ""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in messages[-4:]
    )