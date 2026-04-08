import asyncio
import hashlib
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)



# Cache key helpers

def make_cache_key(prompt: str, model: str, temperature: float, max_tokens: int) -> str:
    """
    Deterministic, privacy-safe cache key.

    The key is a SHA-256 digest of the canonical JSON of the request
    parameters.  Plain-text user content is never stored – only its hash
    participates in the key.  User identifiers are explicitly excluded.
    """
    key_data = {
        "prompt_hash": hashlib.sha256(prompt.strip().encode()).hexdigest(),
        "model": model,
        # Only cache temperature=0 responses (deterministic); for temp>0
        # the caller should bypass the cache.
        "temperature": round(temperature, 4),
        "max_tokens": max_tokens,
    }
    content = json.dumps(key_data, sort_keys=True)
    digest = hashlib.sha256(content.encode()).hexdigest()
    return f"llm:{digest}"



# In-process LRU + TTL cache


@dataclass
class _CacheEntry:
    value: str
    expires_at: float  # Unix timestamp


class InMemoryCache:
    """
    Thread-safe (asyncio-safe) in-process cache with:
    - TTL-based expiration
    - LRU eviction when max_entries is reached
    - No plaintext user data stored (keys are hashes)
    """

    def __init__(self, ttl_seconds: int = 3600, max_entries: int = 1000):
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

        # Metrics
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    #Public API 

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                # Expired – remove and report miss
                del self._store[key]
                self._misses += 1
                logger.debug("Cache expired: %s", key[:20])
                return None
            # Move to end (most-recently-used)
            self._store.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = _CacheEntry(
                value=value,
                expires_at=time.time() + self._ttl,
            )
            # Evict oldest entries if we exceed max_entries
            while len(self._store) > self._max:
                evicted_key, _ = self._store.popitem(last=False)
                self._evictions += 1
                logger.debug("Cache evicted (LRU): %s", evicted_key[:20])

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
            logger.info("Cache cleared")

    async def purge_expired(self) -> int:
        """Remove all expired entries; returns count removed."""
        now = time.time()
        async with self._lock:
            expired = [k for k, v in self._store.items() if now > v.expires_at]
            for k in expired:
                del self._store[k]
            return len(expired)

    #Metrics 

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend": "memory",
            "size": len(self._store),
            "max_entries": self._max,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
            "evictions": self._evictions,
        }



# Redis-backed cache (optional)

class RedisCache:
    

    def __init__(self, redis_url: str, ttl_seconds: int = 3600, max_entries: int = 1000):
        try:
            import redis.asyncio as aioredis  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Install redis: pip install 'redis[asyncio]'"
            ) from e

        self._client = aioredis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl_seconds
        self._max = max_entries
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Optional[str]:
        value = await self._client.get(key)
        if value is None:
            self._misses += 1
        else:
            self._hits += 1
        return value

    async def set(self, key: str, value: str) -> None:
        await self._client.setex(key, self._ttl, value)

    async def delete(self, key: str) -> bool:
        result = await self._client.delete(key)
        return result > 0

    async def clear(self) -> None:
        await self._client.flushdb()

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "backend": "redis",
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total else 0.0,
        }



# Factory


def build_cache(backend: str, redis_url: str, ttl_seconds: int, max_entries: int):
    """Return the appropriate cache implementation based on configuration."""
    if backend == "redis":
        logger.info("Using Redis cache: %s", redis_url)
        return RedisCache(redis_url, ttl_seconds, max_entries)
    logger.info("Using in-process memory cache (max=%d, ttl=%ds)", max_entries, ttl_seconds)
    return InMemoryCache(ttl_seconds, max_entries)