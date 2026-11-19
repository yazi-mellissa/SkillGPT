import logging

from openai import OpenAI

from app.services.cache import ResponseCache, cache_key

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Extract the professional skills named in the user's text. "
    "Return them as a comma-separated list and nothing else. "
    "Only list skills the text actually mentions; if it mentions none, return an empty line."
)


class OpenAIService:
    """Skill extraction through an OpenAI chat model, with a Redis response cache."""

    def __init__(self, settings):
        self.client = OpenAI(api_key=settings.OPEN_API_KEY)
        self.model_name = settings.OPENAI_MODEL_NAME
        self.generation_config = {
            "max_tokens": 200,
            "n": 1,
            # Temperature 0 on purpose. Extraction is not a creative task, and a
            # deterministic answer is what makes the response cache sound: the cached
            # reply is the one the model would give again.
            "temperature": 0.0,
        }
        self.cache = (
            ResponseCache(settings.REDIS_HOST, settings.REDIS_PORT, settings.REDIS_PASSWORD)
            if settings.USE_GPTCACHE
            else None
        )
        logger.info("OpenAIService ready, model=%s, cache=%s",
                    self.model_name, bool(self.cache and self.cache.available))

    def search(self, query: str) -> list:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        key = cache_key("openai", self.model_name, query)
        if self.cache:
            hit = self.cache.get(key)
            if hit is not None:
                return hit

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            **self.generation_config,
        )
        content = response.choices[0].message.content or ""
        skills = [s.strip() for s in content.split(",") if s.strip()]

        if self.cache:
            self.cache.set(key, skills)
        logger.info("OpenAI returned %d skills", len(skills))
        return skills
