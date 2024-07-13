import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import logging

class SkillBERT:
    """
    A class to interface with BERT for skill extraction.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """
        Initializes a new instance of the SkillBERT class.
        
        Args:
            model_name (str, optional): Name of the pre-trained BERT model to use. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        logging.info(f"SkillBERT instance created with model: {model_name}")

    def _tokenize(self, text: str):
        """
        Tokenizes the input text using the BERT tokenizer.
        
        Args:
            text (str): The text to tokenize.
        
        Returns:
            dict: A dictionary containing tokenized input ids and attention mask.
        """
        tokens = self.tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )
        logging.info("Text tokenized successfully.")
        return tokens

    def get_embeddings(self, text: str):
        """
        Extracts embeddings from the input text using the BERT model.
        
        Args:
            text (str): The text to extract embeddings from.
        
        Returns:
            torch.Tensor: A tensor containing the embeddings.
        """
        tokens = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(**tokens)

        embeddings = outputs.last_hidden_state[:, 0, :]
        logging.info("Embeddings extracted successfully.")
        return embeddings

    def search(self, query: str):
        """
        Searches for relevant skills using the BERT model based on the input query.
        
        Args:
            query (str): The query string to search for.
        
        Returns:
            dict: A dictionary of skills extracted from the embeddings.
        """
        try:
            embeddings = self.get_embeddings(query)
            # Here you would implement the logic to extract relevant skills from the embeddings
            # For the sake of the example, we'll just simulate returning some skills
            logging.info("Successfully retrieved and processed skills.")
            return {}
        except Exception as e:
            logging.error(f"Error retrieving or processing embeddings: {e}")
            raise ValueError(f"Error retrieving or processing embeddings: {e}")

