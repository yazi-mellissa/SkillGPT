import logging

import google.generativeai as genai

from app.services.cache import ResponseCache, cache_key
from app.services.openai_service import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class GeminiService:
    """The same extraction task through Google Gemini, so the two providers compare.

    The prompt is imported from the OpenAI service rather than rewritten. Comparing two
    providers on two different prompts measures the prompts.
    """

    def __init__(self, settings):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model_name = settings.GEMINI_MODEL_NAME
        self.generation_config = {
            "temperature": 0.0,
            "max_output_tokens": 512,
            "response_mime_type": "text/plain",
        }
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
            system_instruction=SYSTEM_PROMPT,
        )
        self.cache = (
            ResponseCache(settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_PASSWORD)
            if settings.USE_GPTCACHE
            else None
        )
        logger.info("GeminiService ready, model=%s, cache=%s",
                    self.model_name, bool(self.cache and self.cache.available))

    def search(self, query: str) -> list:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        key = cache_key("gemini", self.model_name, query)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        response = self.model.generate_content(query)
        content = (response.text or "")
        skills = [s.strip() for s in content.split(",") if s.strip()]

        if self.cache:
            self.cache.set(key, skills)
        logger.info("Gemini returned %d skills", len(skills))
        return skills
