import openai
from ai_search import SkillAI


class SkillOpenAI(SkillAI):
    def __init__(self, api_key: str, model_name: str, generation_config: dict):
        super().__init__(api_key)
        self.model_name = model_name
        self.generation_config = generation_config

    def search(self, query: str):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=query,
            **self.generation_config
        )
        return response.choices[0].text.strip()