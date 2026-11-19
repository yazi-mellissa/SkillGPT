"""Service instances, built once.

Why this module exists. The routes used to declare their dependencies as
`milvus_service: MilvusMemoryService = Depends()`, which makes FastAPI construct the
class on every request. Each of those constructors does real work: opening a Zilliz
connection, building an embedding client, and in the OpenAI case building a whole
GPTCache manager. So every call paid that cost again, and the cache was rebuilt from
scratch each time, which meant it could never return a hit. The caching feature was
present in the code and absent in behaviour.

Services are therefore built once, on first use, and shared. They hold network clients
and no per-request state, which is what makes sharing them safe.
"""

import logging
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.gemini_service import GeminiService
from app.services.milvus_service import MilvusMemoryService
from app.services.openai_service import OpenAIService

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_milvus_service() -> MilvusMemoryService:
    return MilvusMemoryService(get_settings())


@lru_cache(maxsize=1)
def get_openai_service() -> OpenAIService:
    return OpenAIService(get_settings())


@lru_cache(maxsize=1)
def get_gemini_service() -> GeminiService:
    return GeminiService(get_settings())


@lru_cache(maxsize=1)
def get_encoder_service():
    """Imported lazily: torch and transformers are a large optional dependency, and a
    deployment that only needs the hosted backends should not have to carry them."""
    from app.services.encoder_service import EncoderService

    return EncoderService(get_settings())


def _transformers_available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(m) for m in ("torch", "transformers"))


def warm_up(settings: Settings) -> dict:
    """Build the services configured for this deployment, at startup rather than on the
    first request, so that a missing key or an unreachable Zilliz fails immediately and
    visibly instead of surfacing as one slow, confusing 500.

    A backend whose credentials are absent is skipped rather than fatal: running with
    only the vector index, or only one provider, is a legitimate deployment.
    """
    status = {}
    plan = [
        ("vector", bool(settings.ZILLIZ_URI and settings.OPEN_API_KEY), get_milvus_service),
        ("openai", bool(settings.OPEN_API_KEY), get_openai_service),
        ("gemini", bool(settings.GEMINI_API_KEY), get_gemini_service),
        ("encoder", _transformers_available(), get_encoder_service),
    ]
    for name, configured, factory in plan:
        if not configured:
            status[name] = "not configured"
            logger.warning("backend %s has no credentials, skipping", name)
            continue
        try:
            factory()
            status[name] = "ready"
        except Exception as exc:  # noqa: BLE001 - startup must report, not crash
            status[name] = f"failed: {type(exc).__name__}"
            logger.error("backend %s failed to initialise: %s", name, exc)
    return status
