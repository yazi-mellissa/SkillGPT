# SkillGPT: three ways to pull skills out of free text

Given a job description, a CV or any piece of free text, return the skills it mentions.

The interesting part is that there is no single right way to do this, so this service implements
**three** and puts them behind the same interface:

| Approach | Endpoint | How it decides |
|---|---|---|
| **Vector search** | `/search` | Embeds the query and retrieves the 15 nearest entries from a Milvus/Zilliz collection of known skills. Bounded to a fixed vocabulary, fast, cheap, no hallucination possible. |
| **OpenAI** | `/search_openai` | Asks a GPT model to extract the skills. Open vocabulary, handles phrasing the index has never seen, costs a call and can invent. |
| **Gemini** | `/search_gemini` | The same, through Google Gemini, so the two providers can be compared on identical input. |

Swapping between them is a change of URL, which is the point: the three can be measured against
each other on the same queries.

Responses are cached with GPTCache backed by Redis, so a repeated or near-identical query does not
pay for a second model call.

## Architecture

```
app/
├── main.py               # FastAPI startup
├── api/
│   └── routes.py         # the three search endpoints + health
├── services/
│   ├── milvus_service.py # Zilliz collection, embedding, nearest-neighbour retrieval
│   ├── openai_service.py # skill extraction through OpenAI
│   ├── gemini_service.py # skill extraction through Gemini
│   ├── gpt_cache.py      # GPTCache over Redis
│   └── memory_manager.py # Zilliz Cloud lifecycle
└── core/
    ├── config.py         # settings, read from the environment only
    └── logging.py
```

Configuration is read entirely from the environment through Pydantic settings. No key, URI or token
is ever written in the code, and `.env` is excluded from version control.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose, for the containerised deployment
- API keys for OpenAI and Google Gemini
- A Zilliz Cloud account, for the vector database

## Environment variables

Copy `.env.example` to `.env` and fill it in:

```
# API server
API_HOST=0.0.0.0
API_PORT=8000

# Redis, used by the cache
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# Model providers
OPEN_API_KEY=
GEMINI_API_KEY=
OPENAI_MODEL_NAME=
GEMINI_MODEL_NAME=

# Caching
USE_GPTCACHE=true

# Zilliz / Milvus
ZILLIZ_URI=
ZILLIZ_TOKEN=
WIPE_MILVUS_ON_START=false
```

## Running it

With Docker Compose, which also brings up Redis:

```bash
docker-compose up -d
```

The API is then on http://localhost:8000, with the interactive documentation on `/docs`.

Manually:

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Endpoints

| Route | Returns |
|---|---|
| `GET /search?query=...` | The 15 closest skills in the vector index |
| `GET /search_openai?query=...` | Skills extracted by the OpenAI model |
| `GET /search_gemini?query=...` | Skills extracted by Gemini |
| `GET /api/health` | Health check |

## What is missing

The three approaches have never been measured against each other on a common annotated set.
Precision, recall, latency and cost per query would settle which one to use when, and that
comparison is the next thing this repository should contain.

## License

MIT, see [LICENSE](LICENSE).
