import google.generativeai as genai
import logging
from fastapi import Depends
from app.core.config import get_settings

logger = logging.getLogger(__name__)

class GeminiService:
    """
    A service class to interface with Google Gemini AI for skill extraction.
    """
    
    def __init__(self, settings = Depends(get_settings)):
        """
        Initializes a new instance of the GeminiService class.
        
        Args:
            settings: Application settings from dependency injection
        """
        genai.configure(api_key=settings.GEMINI_API_KEY)
        
        self.generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        self.model_name = settings.GEMINI_MODEL_NAME
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
        )
        
        logger.info(f"GeminiService instance created with model: {self.model_name}")

    def search(self, query: str):
        """
        Searches for relevant skills using Google Gemini based on the input query.
        
        Args:
            query (str): The query string to search for.
        
        Returns:
            list: A list of skills extracted from the response.
        """
        try:
            response = self.model.generate_content([
                f"Extract key skills from the following text and return them as a comma-separated list:",
                f"Text: {query}",
                "Skills: ",
            ])
            
            content = response.text
            skills_list = [skill.strip() for skill in content.split(',')]
            
            logger.info(f"Successfully extracted {len(skills_list)} skills using Gemini model")
            return skills_list
            
        except Exception as e:
            logger.error(f"Error extracting skills with Gemini: {e}")
            raise