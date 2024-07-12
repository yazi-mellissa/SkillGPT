import requests

class SkillGPT:
    def __init__(self, api_key):
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        self.model_id = "pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
        self.api_url = f"https://api-inference.huggingface.co/{self.model_id}"

    def generate_text(self, prompt):
        data = {"inputs": prompt}

        response = requests.post(self.api_url, headers=self.headers, json=data)
        if response.status_code == 200:
            return response.json()
        else:
            return response.json().get('error', 'An error occurred')
