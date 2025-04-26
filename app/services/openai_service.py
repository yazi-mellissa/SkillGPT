from openai import OpenAI
import logging
from fastapi import Depends
from app.services.gpt_cache import GPTCacheManager
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class OpenAIService:
    """
    A service class to interface with OpenAI models for skill extraction.
    """
    
    def __init__(self, settings = Depends(get_settings)):
        """
        Initializes the OpenAIService with OpenAI API key, model name, and generation configuration.
        
        Args:
            settings: Application settings from dependency injection
        """
        self.original_client = OpenAI(api_key=settings.OPEN_API_KEY)
        self.model_name = settings.OPENAI_MODEL_NAME
        
        self.generation_config = {
            "max_tokens": 100,
            "n": 1,
            "stop": None,
            "temperature": 0.5,
        }
        
        self.use_cache = settings.USE_GPTCACHE
        
        if self.use_cache:
            self.cache_manager = GPTCacheManager(
                settings.OPEN_API_KEY, 
                settings.ZILLIZ_URI, 
                settings.ZILLIZ_TOKEN
            )
            self.client = self.cache_manager.get_cached_client(self.original_client)
        else:
            self.client = self.original_client
            
        logger.info(f"OpenAIService instance created with model: {self.model_name}, cache enabled: {self.use_cache}")

    def search(self, query: str):
        """
        Searches for skills using the OpenAI API based on the given query.
        
        Args:
            query (str): User input query to search for.
        
        Returns:
            list: A list of skills extracted from the response.
        """
        if not query:
            logger.error("Empty query provided")
            raise ValueError("Query cannot be empty")

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Extract key skills from the user's input and return them as a comma-separated list."},
                    {"role": "user", "content": query},
                ],
                **self.generation_config
            )
            
            content = response.choices[0].message.content
            skills_list = [skill.strip() for skill in content.split(',')]

            logger.info(f"Successfully extracted {len(skills_list)} skills using OpenAI model")
            return skills_list
        
        except Exception as e:
            logger.error(f"Failed to extract skills with OpenAI: {e}")
            raise