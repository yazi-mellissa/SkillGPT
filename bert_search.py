import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import logging


# Link to the google colab : https://colab.research.google.com/drive/1S9CaidSbCmetlsjKoSTLcATej0D3i_P0#scrollTo=hLciQaKm_2Qx
class SkillBERT:
    """
    A class to interface with BERT for skill extraction.
    """
    
    def __init__(self, model_path: str = 'bert-base-uncased'):
        """
        Initializes a new instance of the SkillBERT class.
        
        Args:
            model_name (str, optional): Name of the pre-trained BERT model to use. Defaults to 'bert-base-uncased'.
        """
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        logging.info(f"SkillBERT instance created with model: {model_path}")

    def _tokenize(self, text: str,max_length=128):
        """
        Tokenizes the input text using the BERT tokenizer.
        
        Args:
            text (str): The text to tokenize.
        
        Returns:
            dict: A dictionary containing tokenized input ids and attention mask.
        """
        tokenized_input = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
        logging.info("Text tokenized successfully.")
        return tokenized_input

    def predict_skills(self, text):
        self.model.eval()  # Set the model to evaluation mode
        inputs = self._tokenize(text, self.tokenizer)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        predicted_labels = predictions.squeeze().tolist()

        skills = []
        for token, label in zip(tokens, predicted_labels):
            if label == 1 and token != '[PAD]':
                skills.append(token)

        return skills
    
    def search(self, query: str):
        """
        Searches for relevant skills using the BERT model based on the input query.
        
        Args:
            query (str): The query string to search for.
        
        Returns:
            dict: A dictionary of skills extracted from the embeddings.
        """
        try:
            skills = self.predict_skills(query)
            logging.info("Successfully retrieved and processed skills.")
            return skills
        except Exception as e:
            logging.error(f"Error retrieving or processing embeddings: {e}")
            raise ValueError(f"Error retrieving or processing embeddings: {e}")