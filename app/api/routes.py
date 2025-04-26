from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any
import logging
from app.services.milvus_service import MilvusMemoryService
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService
from app.core.config import get_settings

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/search", response_model=List[Dict[str, Any]])
async def search(query: str, milvus_service: MilvusMemoryService = Depends()):
    """
    Search for skills using vector similarity in Milvus/Zilliz
    """
    try:
        query_vectors = milvus_service.encode_query(query)
        results = milvus_service.get_relevant(query_vectors=query_vectors, num_relevant=15)
        return results
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/search_gemini", response_model=List[str])
async def search_gemini(query: str, gemini_service: GeminiService = Depends()):
    """
    Search for skills using Google Gemini AI
    """
    try:
        skills = gemini_service.search(query)
        return skills
    except Exception as e:
        logger.error(f"Error in search_gemini endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini search failed: {str(e)}")

@router.get("/search_openai", response_model=List[str])
async def search_openai(query: str, openai_service: OpenAIService = Depends()):
    """
    Search for skills using OpenAI models
    """
    try:
        skills = openai_service.search(query)
        return skills
    except Exception as e:
        logger.error(f"Error in search_openai endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI search failed: {str(e)}")
  