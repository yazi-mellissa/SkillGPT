import os
import argparse
import uvicorn
from fastapi import FastAPI, Request
from dotenv import load_dotenv
from milvus_search import MilvusMemory
from gemini_search import SkillGemini
from open_ai_search import SkillOpenAI
from bert_search import SkillBERT

app = FastAPI()

load_dotenv()
API_HOST = os.getenv("API_HOST")
API_PORT = int(os.getenv("API_PORT"))
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
OPEN_API_KEY = os.getenv("OPEN_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


@app.get("/search_rdsi_bkylm")
def search(query: str):
    query_vectors = milvus_memory.embedding_fn.encode_queries([query])
    return milvus_memory.get_relevant(collection="skills",query_vectors=query_vectors, num_relevant=15)

@app.get("/search_gemini")
def search(query: str):
    return skillGemini.search(query)

@app.get("/search_openai")
def search(query: str):
    skills = skillOpenAI.search(query)
    return skills
    
@app.get("/search_bert")
def search(query: str):
    skills = skillBERT.search(query)
    return skills


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--memory-backend", type=str, default="redis")
    parser.add_argument("--redis-host", type=str, default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    parser.add_argument("--redis-password", type=str, default=REDIS_PASSWORD)
    parser.add_argument("--open-api-key", type=str, default=OPEN_API_KEY)
    parser.add_argument("--gemini_api-key", type=str, default=GEMINI_API_KEY)
    args = parser.parse_args()

    milvus_memory = MilvusMemory("milvus_demo.db",args.open_api_key , wipe_milvus_on_start=False)

    # from get code in ai.studio google.generativeai import genai
    generation_config_gemini = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 0,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
    }
    model_name_gemini = "tunedModels/rdsi-hsrs1unlkdaz"

    generation_config_open_ai = {
        "max_tokens": 100,
        "n":1,
        "stop":None,
        "temperature":0.5,
    }

    model_name_open_ai = "ft:gpt-3.5-turbo-0125:rdsi:expertise1:9kScjB8w"
    skillGemini = SkillGemini(api_key=args.gemini_api_key, model_name=model_name_gemini ,generation_config=generation_config_gemini)
    skillOpenAI = SkillOpenAI(api_key=args.open_api_key, model_name=model_name_open_ai ,generation_config=generation_config_open_ai)
    skillBERT = SkillBERT("models/fine-tuned-bert")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")  