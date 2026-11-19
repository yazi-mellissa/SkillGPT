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

## Benchmark

Three interchangeable backends are only useful if you can say which one to use when, so
[`benchmark/`](benchmark/) measures them against each other on the same input.

```bash
python -m benchmark.run_benchmark --url http://localhost:8000
python -m benchmark.run_benchmark --mock      # no service or API key needed
pytest benchmark/
```

**The annotated set.** [`dataset.jsonl`](benchmark/dataset.jsonl) holds 30 queries with their
expected skills: 15 job descriptions, 10 CV extracts, and 5 deliberately awkward cases. The
awkward ones are where backends separate: a two-word query with nothing to go on, a sentence
that *negates* a skill ("you will not need any frontend experience"), a paragraph about the
office with no skills in it at all, and one query in French.

**What is measured, and why not a single score.**

| | |
|---|---|
| precision, recall, F1 | reported separately, macro and micro |
| latency | p50 and p95, which is what a caller feels |
| cost | dollars per thousand queries, zero for the vector index |

Precision and recall are kept apart on purpose. The vector backend cannot invent a skill, so
its precision is bounded by its vocabulary and its recall by whatever is not in the index. An
LLM can name a skill the index has never seen, and can also name one that is not in the text.
A single F1 ranks a backend that invents skills above one that misses them; whether that is the
right ranking depends on what the caller does with the answer, so the table shows both and
leaves the choice where it belongs.

**Normalisation.** The three endpoints return three shapes, and "Node.js", "NodeJS" and "node js"
are one skill. [`normalize.py`](benchmark/normalize.py) folds all of that before scoring, because
otherwise the benchmark measures formatting rather than extraction. The alias table is kept
deliberately short: every entry in it makes the scores look better, so each one has to be a
spelling of the same skill and not merely a related one.

**Caching.** The service caches responses, so a second run measures the cache. Either start it
with `USE_GPTCACHE=false` or pass `--cache-warmup`, which queries everything once before timing
so that all three backends are measured in the same state. Comparing a cold LLM against a warm
index is the easiest way to get a wrong answer out of this script.

`--mock` runs the whole harness against synthetic responses with a known failure mode per
backend, which is how the scoring itself is tested: `pytest benchmark/` covers the metrics and
the normalisation, the two pieces that decide every number the table prints.

**Results are not committed.** Running the benchmark costs API calls, and the numbers depend on
which models are configured and on the contents of the Zilliz collection, so publishing one run
as though it were the answer would be misleading. Run it against your own deployment.

## License

MIT, see [LICENSE](LICENSE).
