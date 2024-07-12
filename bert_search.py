import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class SkillBERT:
    def __init__(self, model_path, tokenizer_path):
        self.model = BertModel.from_pretrained(model_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.model.eval()
        self.model.to(device)
        
    def encode(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0).to(device)
        with torch.no_grad():
            last_hidden_states = self.model(input_ids)[0]
        return last_hidden_states

    def search(self, query, skills, top_k=5):
        query_vec = self.encode(query)
        skill_vecs = [self.encode(skill) for skill in skills]
        scores = [(i, F.cosine_similarity(query_vec, skill_vec.unsqueeze(0)).item()) for i, skill_vec in enumerate(skill_vecs)]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        return scores[:top_k]