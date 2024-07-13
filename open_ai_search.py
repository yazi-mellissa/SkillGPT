from openai import OpenAI
import logging
from ai_search import SkillAI

class SkillOpenAI(SkillAI):
    def __init__(self, api_key: str, model_name: str, generation_config: dict):
        """
        Initializes the SkillOpenAI object with OpenAI API key, model name, and generation configuration.
        
        Args:
            api_key (str): OpenAI API key.
            model_name (str): Name of the model to use for generation.
            generation_config (dict): Configuration dictionary for generating responses.
        """
        super().__init__(api_key)
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.generation_config = generation_config

    def search(self, query: str):
        """
        Searches for an answer using the OpenAI API based on the given query.
        
        Args:
            query (str): User input query to search for.
        
        Returns:
            str: The content of the response from the assistant.
        """
        if not query:
            logging.error("Empty query provided.")
            return "Query cannot be empty."

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "System initialized to extract and respond with skills from a predefined list."},
                    {"role": "user", "content": query},
                ],
                **self.generation_config
            )
            content = response.choices[0].message.content
            skills_list = [skill.strip() for skill in content.split(',')]

            logging.info("Successfully retrieved response.")
            return skills_list
        
        except Exception as e:
            logging.error(f"Failed to retrieve response: {e}")
            return "An error occurred while processing your request."
