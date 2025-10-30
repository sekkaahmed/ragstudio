"""
Redis Cache for RAG Studio - OPTIONAL FEATURE (Currently Unused)

⚠️ STATUS: This module is prepared but NOT currently integrated into the pipeline.
Redis is an optional performance optimization that can be enabled in the future.

The application works fully without Redis. This module provides a ready-to-use
caching layer when you want to optimize performance for production workloads.

FUTURE USE CASES:
- Cache chunks (avoid re-chunking identical documents)
- Cache embeddings (expensive GPU/API calls)
- Cache search results (frequent queries)

TO ENABLE:
1. Install Redis: `brew install redis` or `docker run -d redis`
2. Instantiate RedisCache in your pipeline
3. Pass it to chunking/embedding functions

CURRENT STATE: Code is present but never instantiated = 0% usage
"""

import json
import hashlib
import logging
from typing import Optional, Any, List
import redis
from functools import wraps
import time


logger = logging.getLogger(__name__)


class RedisCache:
    """Redis-based cache for Atlas-RAG"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600,  # 1 hour
        enabled: bool = True
    ):
        """
        Initialize Redis cache

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            default_ttl: Default TTL in seconds
            enabled: Enable/disable caching
        """
        self.enabled = enabled
        self.default_ttl = default_ttl

        if not enabled:
            logger.info("Redis cache disabled")
            self.client = None
            return

        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache connected: {host}:{port}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Cache disabled.")
            self.enabled = False
            self.client = None

    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generate cache key from arguments

        Args:
            prefix: Key prefix (e.g., 'chunk', 'embed', 'search')
            *args: Positional arguments to hash
            **kwargs: Keyword arguments to hash

        Returns:
            Cache key string
        """
        # Create a string representation of all arguments
        key_data = f"{args}:{sorted(kwargs.items())}"
        # Hash it for consistent key length (not for security)
        key_hash = hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()
        return f"atlas:{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if not self.enabled or not self.client:
            return None

        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache HIT: {key}")
                return json.loads(value)
            logger.debug(f"Cache MISS: {key}")
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        if not self.enabled or not self.client:
            return False

        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value, ensure_ascii=False)
            self.client.setex(key, ttl, serialized)
            logger.debug(f"Cache SET: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self.enabled or not self.client:
            return False

        try:
            self.client.delete(key)
            logger.debug(f"Cache DELETE: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False

    def clear_prefix(self, prefix: str) -> int:
        """
        Clear all keys with given prefix

        Args:
            prefix: Key prefix to clear

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.client:
            return 0

        try:
            pattern = f"atlas:{prefix}:*"
            keys = self.client.keys(pattern)
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys with prefix '{prefix}'")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return 0

    def get_stats(self) -> dict:
        """Get cache statistics"""
        if not self.enabled or not self.client:
            return {"enabled": False}

        try:
            info = self.client.info("stats")
            return {
                "enabled": True,
                "total_keys": self.client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "hit_rate": (
                    info.get("keyspace_hits", 0) /
                    (info.get("keyspace_hits", 0) + info.get("keyspace_misses", 1))
                )
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"enabled": True, "error": str(e)}

    # High-level cache methods

    def cache_chunks(
        self,
        text: str,
        strategy: str,
        max_tokens: int,
        overlap: int,
        chunks: List[dict],
        ttl: int = 3600
    ) -> bool:
        """
        Cache chunking results

        Args:
            text: Input text
            strategy: Chunking strategy
            max_tokens: Max tokens parameter
            overlap: Overlap parameter
            chunks: Resulting chunks
            ttl: Time to live

        Returns:
            True if cached
        """
        key = self._make_key(
            "chunk",
            text=text[:100],  # Use first 100 chars as part of key
            strategy=strategy,
            max_tokens=max_tokens,
            overlap=overlap
        )
        return self.set(key, chunks, ttl)

    def get_cached_chunks(
        self,
        text: str,
        strategy: str,
        max_tokens: int,
        overlap: int
    ) -> Optional[List[dict]]:
        """Get cached chunks"""
        key = self._make_key(
            "chunk",
            text=text[:100],
            strategy=strategy,
            max_tokens=max_tokens,
            overlap=overlap
        )
        return self.get(key)

    def cache_search_results(
        self,
        query: str,
        top_k: int,
        results: List[dict],
        ttl: int = 300  # 5 minutes for search results
    ) -> bool:
        """Cache search results"""
        key = self._make_key("search", query=query, top_k=top_k)
        return self.set(key, results, ttl)

    def get_cached_search(
        self,
        query: str,
        top_k: int
    ) -> Optional[List[dict]]:
        """Get cached search results"""
        key = self._make_key("search", query=query, top_k=top_k)
        return self.get(key)

    def cache_embeddings(
        self,
        text: str,
        model: str,
        embedding: List[float],
        ttl: int = 86400  # 24 hours
    ) -> bool:
        """Cache embeddings"""
        key = self._make_key("embed", text=text, model=model)
        return self.set(key, embedding, ttl)

    def get_cached_embedding(
        self,
        text: str,
        model: str
    ) -> Optional[List[float]]:
        """Get cached embedding"""
        key = self._make_key("embed", text=text, model=model)
        return self.get(key)


def cached(cache_instance: RedisCache, ttl: Optional[int] = None):
    """
    Decorator for caching function results

    Usage:
        @cached(cache, ttl=3600)
        def expensive_function(arg1, arg2):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache_instance._make_key(
                func.__name__,
                *args,
                **kwargs
            )

            # Try to get from cache
            cached_result = cache_instance.get(key)
            if cached_result is not None:
                return cached_result

            # Call function
            result = func(*args, **kwargs)

            # Cache result
            cache_instance.set(key, result, ttl)

            return result
        return wrapper
    return decorator
