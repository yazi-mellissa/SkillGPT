import logging

class SkillAI:
    def __init__(self, api_key: str, api_url: str = None):
        """
        Initializes a new instance of the SkillAI class.
        
        Args:
            api_key (str): The API key for authentication.
            api_url (str, optional): The base URL of the API service. Defaults to None.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.auth = {"Authorization": f"Bearer {api_key}"}
        logging.info("SkillAI instance created with API key and URL.")
    
    def search(self, query: str):
        """
        Abstract method to perform a search query. This method should be overridden in subclasses.
        
        Args:
            query (str): The query string for which to search.
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    