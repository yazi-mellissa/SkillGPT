import torch
from transformers import BertForSequenceClassification, BertTokenizer
import logging
from ISearch import ISearch

class SkillBERT(ISearch):
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
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)  # Assuming binary classification (skill or not skill)
        logging.info(f"SkillBERT instance created with model: {model_path}")

    def _tokenize(self, text: str, max_length=128):
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
        self.model.eval()
        inputs = self._tokenize(text)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        logging.info(f"Logits: {logits}")
        predictions = torch.argmax(logits, dim=-1)
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Predictions shape: {predictions.shape}")

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze().tolist())
        logging.info(f"Tokens: {tokens}")

        if predictions.dim() == 0:
            predicted_labels = [predictions.item()] * len(tokens)
        else:
            predicted_labels = predictions.squeeze().tolist()

        logging.info(f"Predicted labels: {predicted_labels}")

        if isinstance(predicted_labels, int):
            predicted_labels = [predicted_labels]

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