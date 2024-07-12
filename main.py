import os
import argparse
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from milvus_search import MilvusMemory


app = FastAPI()

load_dotenv()
API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = os.getenv("REDIS_PORT")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
API_KEY = os.getenv("API_KEY")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/search")
def search(query: str):
    query_vectors = milvus_memory.embedding_fn.encode_queries([query])
    return milvus_memory.get_relevant(collection="skills",query_vectors=query_vectors, num_relevant=15)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=API_HOST)
    parser.add_argument("--port", type=int, default=API_PORT)
    parser.add_argument("--memory-backend", type=str, default="redis")
    parser.add_argument("--redis-host", type=str, default=REDIS_HOST)
    parser.add_argument("--redis-port", type=int, default=REDIS_PORT)
    parser.add_argument("--redis-password", type=str, default=REDIS_PASSWORD)
    parser.add_argument("--api-key", type=str, default=API_KEY)
    args = parser.parse_args()

    milvus_memory = MilvusMemory("milvus_demo.db", wipe_milvus_on_start=False)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")