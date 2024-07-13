import google.generativeai as genai
import logging
from ISearch import ISearch

#TODO: Fix the SkillGemini solution
class SkillGemini(ISearch):
    """
    A class to interface with Google Gemini AI for skill extraction.
    """
    
    def __init__(self, api_key: str, model_name: str, generation_config: dict):
        """
        Initializes a new instance of the SkillGemini class.
        
        Args:
            api_key (str): API key for Google Gemini.
            model_name (str): Name of the model to use.
            generation_config (dict): Configuration dictionary for generation settings.
        """
        genai.configure(api_key=api_key)
        self.generation_config = generation_config
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
        )
        logging.info("SkillGemini instance created and configured with API key and model.")

    def search(self, query: str):
        """
        Searches for relevant skills using Google Gemini based on the input query.
        
        Args:
            query (str): The query string to search for.
        
        Returns:
            dict: A dictionary of skills extracted from the response.
        """
        try:
            response = self.model.generate_content([
                f"input: {query}",
                "output: ",
            ])
            content = response['output']
            skills_list = [skill.strip() for skill in content.split(',')]
            logging.info("Successfully retrieved and processed skills.")
            return skills_list
        except Exception as e:
            logging.error(f"Error retrieving or processing response: {e}")
            return {}