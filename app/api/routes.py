import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.deps import (
    get_encoder_service,
    get_gemini_service,
    get_milvus_service,
    get_openai_service,
)
from app.services.gemini_service import GeminiService
from app.services.milvus_service import MilvusMemoryService
from app.services.openai_service import OpenAIService

router = APIRouter()
logger = logging.getLogger(__name__)

# The three endpoints answer the same question through different machinery, so they take
# the same argument and differ only in what they are willing to return.
QUERY = Query(..., min_length=1, max_length=20_000,
              description="Free text: a job description, a CV, or a fragment of either.")


@router.get("/search", response_model=list[dict[str, Any]],
            summary="Nearest skills in the vector index")
async def search(
    query: str = QUERY,
    milvus_service: MilvusMemoryService = Depends(get_milvus_service),
):
    """Embed the query and return the closest entries of a fixed skill vocabulary.

    Bounded to what the index contains, so it cannot invent a skill, and cannot find one
    the index has never been given.
    """
    try:
        vectors = milvus_service.encode_query(query)
        return milvus_service.get_relevant(query_vectors=vectors, num_relevant=15)
    except Exception as exc:  # noqa: BLE001
        logger.exception("vector search failed")
        raise HTTPException(status_code=502, detail=f"Vector search failed: {exc}") from exc


@router.get("/search_openai", response_model=list[str],
            summary="Skills extracted by an OpenAI model")
async def search_openai(
    query: str = QUERY,
    openai_service: OpenAIService = Depends(get_openai_service),
):
    """Open vocabulary: handles phrasings the index has never seen, and can invent."""
    try:
        return openai_service.search(query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OpenAI search failed")
        raise HTTPException(status_code=502, detail=f"OpenAI search failed: {exc}") from exc


@router.get("/search_gemini", response_model=list[str],
            summary="Skills extracted by Google Gemini")
async def search_gemini(
    query: str = QUERY,
    gemini_service: GeminiService = Depends(get_gemini_service),
):
    """The same task through a second provider, so the two can be compared on one input."""
    try:
        return gemini_service.search(query)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Gemini search failed")
        raise HTTPException(status_code=502, detail=f"Gemini search failed: {exc}") from exc


@router.get("/search_encoder", response_model=list[dict[str, Any]],
            summary="Skills matched by a local transformer encoder")
async def search_encoder(
    query: str = QUERY,
    top_k: int = Query(15, ge=1, le=100),
    min_score: float = Query(0.0, ge=-1.0, le=1.0,
                             description="Drop matches below this cosine similarity."),
    encoder=Depends(get_encoder_service),
):
    """BERT, RoBERTa or DistilBERT running locally, chosen by ENCODER_MODEL.

    Free per query and offline, and bounded by the same catalogue as /search, so the two
    differ only in how they embed. Each match carries its similarity, so a caller can
    trade precision against recall without a second request.
    """
    try:
        return encoder.search(query, top_k=top_k, min_score=min_score)
    except Exception as exc:  # noqa: BLE001
        logger.exception("encoder search failed")
        raise HTTPException(status_code=502, detail=f"Encoder search failed: {exc}") from exc
