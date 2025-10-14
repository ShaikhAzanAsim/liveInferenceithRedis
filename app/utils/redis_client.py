import redis.asyncio as redis
from app.config import REDIS_URL

_redis: redis.Redis | None = None

def get_redis() -> redis.Redis:
    global _redis
    if _redis is None:
        _redis = redis.from_url(REDIS_URL, decode_responses=False)  # store bytes
    return _redis
