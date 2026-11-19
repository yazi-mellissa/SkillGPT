import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings
from app.core.deps import warm_up
from app.core.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# What each backend reported at startup. Exposed on /api/health so that a deployment
# missing a key is visible from outside rather than only in the logs.
BACKEND_STATUS: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the services once, before the first request.

    `@app.on_event` is deprecated in the FastAPI version this project depends on; this is
    the supported replacement. Doing the work here rather than lazily also means a bad
    configuration fails at startup, where it is obvious, instead of as the first user's
    unexplained error.
    """
    settings = get_settings()
    logger.info("starting on %s:%s", settings.API_HOST, settings.API_PORT)
    BACKEND_STATUS.update(warm_up(settings))
    logger.info("backends: %s", BACKEND_STATUS)
    yield
    logger.info("shutting down")


app = FastAPI(
    title="SkillGPT",
    description="Skill extraction from free text through three interchangeable backends.",
    version="1.1.0",
    lifespan=lifespan,
)
app.include_router(router)


@app.get("/api/health", summary="Liveness and per-backend readiness")
def health_check():
    """Healthy means the process is up. `backends` says which of the three can answer.

    The two are reported separately on purpose: a container orchestrator should not
    restart the service because one provider's key is missing.
    """
    return {"status": "healthy", "backends": BACKEND_STATUS}
