from abc import ABC, abstractmethod

class ISearch(ABC):
    @abstractmethod
    def search(self, query: str):
        """
        Abstract method to perform a search query. This method should be overridden in subclasses.
        
        Args:
            query (str): The query string for which to search.
        
        Raises:
            NotImplementedError: If the subclass does not override this method.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
    