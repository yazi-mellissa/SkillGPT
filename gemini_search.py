import google.generativeai as genai


class SkillAI:
    def __init__(self, api_key:str, api_url:str):
        self.api_key = api_key
        self.auth = {"Authorization": f"Bearer {api_key}"}
        self.api_url = api_url
    
    def __init__(self, api_key:str):
        self.api_key = api_key
        self.auth = {"Authorization": f"Bearer {api_key}"}

    def search(self, query:str):
        pass


class SkillGemini(SkillAI):
    def __init__(self, api_key:str,model_name:str,generation_config:dict):
        super().__init__(api_key)
        genai.configure(api_key=self.api_key)
        self.generation_config = generation_config
        self.model_name = model_name
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=self.generation_config,
        )

    def search(self, query: str):
        response = self.model.generate_content([
        f"input: {query}",
        "output: ",
        ])
        print(response)
        return response