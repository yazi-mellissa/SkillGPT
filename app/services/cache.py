"""Response cache, backed by Redis.

This replaces GPTCache. The reason is not preference: `gptcache.adapter.openai` subclasses
`openai.ChatCompletion`, which was removed in openai 1.0, while this project depends on
openai>=1.3. Importing the old cache module therefore raised at import time, and since
the OpenAI service imported it unconditionally, the whole application failed to start.
The caching that the README advertised had never run.

What replaces it is deliberately smaller. An exact-match cache on a normalized query
covers the case that actually repeats in this service, the same advert or CV submitted
twice, and it is cheap enough to reason about: a hit costs one Redis round trip and no
embedding call. Semantic caching, where a *similar* query returns a stored answer, is a
different and riskier feature; it is not implemented here rather than implemented badly.
"""

import hashlib
import json
import logging
import re
import unicodedata

logger = logging.getLogger(__name__)

_WHITESPACE = re.compile(r"\s+")


def cache_key(backend: str, model: str, query: str) -> str:
    """Key on the backend, the model and the normalized query.

    The model is part of the key on purpose. Without it, switching OPENAI_MODEL_NAME
    would keep serving answers produced by the previous model, which is the kind of bug
    that survives for months because nothing errors.
    """
    text = unicodedata.normalize("NFKC", query).strip().lower()
    text = _WHITESPACE.sub(" ", text)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    return f"skillgpt:{backend}:{model}:{digest}"


class ResponseCache:
    """A small wrapper over Redis that never raises into the request path.

    A cache that can break the service it accelerates is worse than no cache, so every
    operation is guarded: if Redis is down, reads miss and writes are dropped, and the
    service keeps answering from the model.
    """

    def __init__(self, host: str, port: int, password: str = "", ttl_seconds: int = 86_400):
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0
        self.errors = 0
        self._client = None
        try:
            import redis

            self._client = redis.Redis(
                host=host, port=port, password=password or None,
                socket_connect_timeout=2, socket_timeout=2,
                decode_responses=True,
            )
            self._client.ping()
            logger.info("response cache connected to redis at %s:%s", host, port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("response cache unavailable (%s), running without it", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def get(self, key: str) -> list | None:
        if self._client is None:
            return None
        try:
            raw = self._client.get(key)
        except Exception:  # noqa: BLE001
            self.errors += 1
            return None
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, value: list) -> None:
        if self._client is None:
            return
        try:
            self._client.setex(key, self.ttl, json.dumps(value))
        except Exception:  # noqa: BLE001
            self.errors += 1

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "available": self.available,
            "hits": self.hits,
            "misses": self.misses,
            "errors": self.errors,
            "hit_rate": round(self.hits / total, 4) if total else None,
        }
