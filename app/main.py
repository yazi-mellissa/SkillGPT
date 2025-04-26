import logging
from fastapi import FastAPI
from app.core.config import get_settings
from app.api.routes import router
from app.core.logging import setup_logging

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Skills API", description="API for searching skills using various AI models")

# Include API routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up the application")
    settings = get_settings()
    logger.info(f"Running with settings: API_HOST={settings.API_HOST}, API_PORT={settings.API_PORT}")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down the application")

@app.get("/api/health")
def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy"}