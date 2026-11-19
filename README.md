# SkillGPT: four ways to pull skills out of free text

Given a job description, a CV or any piece of free text, return the skills it mentions.

There is no single right way to do this, so this service implements **four** and puts them behind
one interface, which is what makes them comparable:

| Approach | Endpoint | How it decides |
|---|---|---|
| **Vector search** | `/search` | Embeds the query and retrieves the 15 nearest entries from a Milvus/Zilliz collection of known skills. Bounded to a fixed vocabulary, fast, cheap, no hallucination possible. |
| **OpenAI** | `/search_openai` | Asks a GPT model to extract the skills. Open vocabulary, handles phrasing the index has never seen, costs a call and can invent. |
| **Gemini** | `/search_gemini` | The same, through Google Gemini, so the two providers can be compared on identical input. |
| **Local encoder** | `/search_encoder` | BERT, RoBERTa or DistilBERT running on the machine itself. Free per query, offline, bounded by the same catalogue as the vector index. Selected with `ENCODER_MODEL`. |

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
| `GET /search_encoder?query=...&top_k=15&min_score=0.9` | Catalogue skills matched by the local encoder, each with its cosine similarity |
| `GET /api/health` | Liveness, plus which of the four backends is actually configured |

## The local encoder, measured

`python -m benchmark.compare_encoders` compares the three encoders on the annotated set.
No service, no API key, no cost: the models run in process, so anyone who clones this
repository can reproduce the table below.

Every model shares the catalogue, the pooling and the scoring, so the differences come
from pre-training alone. `spans` scores every one-to-three-word window of the query
against the catalogue and keeps each skill's best window. `document` embeds the whole
query as a single vector, which is the obvious approach.

| Backend | macro F1 | precision | recall | p50 ms |
|---|---|---|---|---|
| **DistilBERT, spans, threshold 0.92** | **0.690** | 0.622 | **0.844** | 645 |
| BERT, spans, threshold 0.92 | 0.685 | **0.736** | 0.723 | 988 |
| DistilBERT, spans, threshold 0.85 | 0.549 | 0.420 | 0.882 | 492 |
| BERT, spans, threshold 0.85 | 0.482 | 0.370 | 0.779 | 900 |
| RoBERTa, spans, any threshold | 0.414 | 0.310 | 0.712 | 971 |
| BERT, document, threshold 0 | 0.115 | 0.083 | 0.258 | 133 |
| DistilBERT, document, threshold 0 | 0.092 | 0.067 | 0.219 | 67 |
| RoBERTa, document, threshold 0 | 0.059 | 0.043 | 0.127 | 91 |

Four things came out of this, and three of them were not what I expected.

**Granularity is worth six times more than the model.** Matching short windows instead of
the whole document takes BERT from 0.115 to 0.685 macro F1. Every model is useless in
`document` mode and usable in `spans` mode. Comparing a 30-word advert to a one-word
skill compares objects of different lengths, and the nearest neighbours come back
dominated by whatever the document is broadly about.

**The smallest model wins.** DistilBERT has 40% fewer parameters than BERT, loads in
2.7 seconds against 40.9, answers in 645 ms against 988, and still scores highest. It
trades precision for recall against BERT rather than being better at everything, but on
this task the trade is favourable and it costs less at every step.

**RoBERTa is the worst of the three, and insensitive to the threshold.** It scores
identically at 0, 0.85 and 0.92, which means its similarities sit in a band narrower than
the sweep. A threshold is the only tuning knob this backend has, and on RoBERTa it does
nothing.

**The threshold is not a detail.** For DistilBERT, moving it from 0 to 0.92 raises macro
F1 from 0.522 to 0.690 at the cost of four points of recall. A nearest-neighbour backend
always returns `top_k` answers, so the threshold is what turns a ranked list into a
decision.

What this does not settle: how a local encoder compares against the hosted models. That
needs API keys, so it is left to whoever has them, through
`python -m benchmark.run_benchmark`.


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

## Continuous integration and deployment

Two pipelines, because they do different jobs.

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on every push to this
repository: lint, then the benchmark tests on Python 3.10, 3.11 and 3.12, then it builds
the image, **starts it with no credentials at all and waits for `/api/health`**. That last
step exists because a published version of this project could not be imported at all and
nothing caught it, for the simple reason that nothing had ever tried to start it.

[`.gitlab-ci.yml`](.gitlab-ci.yml) is the shipping path: the same checks, then push to a
container registry and roll out to Azure Container Apps, with the deploy stage manual so a
release is a decision rather than a consequence of merging.

### Azure

[`deploy/azure/main.bicep`](deploy/azure/main.bicep) describes the whole environment:
Container Apps, Azure Cache for Redis, Log Analytics, ingress, and every key as a secret
reference rather than a plain environment value.

```bash
export REGISTRY_USER=... REGISTRY_PASSWORD=... OPEN_API_KEY=... GEMINI_API_KEY=...
./deploy/azure/deploy.sh my-resource-group westeurope myregistry.azurecr.io/skillgpt:v1
```

Two choices worth explaining. **Container Apps rather than AKS**, because this is one
stateless container that should cost nothing between recruitment campaigns; `minReplicas`
is 0 and it scales on HTTP concurrency. And **the liveness probe is not a readiness
probe**: `/api/health` reports a missing provider key as a degraded backend, and a service
answering on two backends out of four should keep taking traffic rather than be pulled out
of rotation for it.


## What was wrong, and what changed

The service as published could not start. These are the defects found and fixed, recorded
because the first one explains a feature that was documented and had never run.

- **The response cache could never return a hit.** The routes declared their dependencies
  as `Depends()` on the service classes, so FastAPI rebuilt each service on every request,
  and each constructor opens network clients. The OpenAI service rebuilt its whole cache
  manager per call, so the cache was empty every time. Services are now built once.
- **The application could not import at all.** `gptcache.adapter.openai` subclasses
  `openai.ChatCompletion`, removed in openai 1.0, while `requirements.txt` asked for
  openai>=1.3. The import raised, and it sat on the path of every request. GPTCache is
  replaced by a small explicit Redis cache.
- **The container crashed on startup.** `docker-compose.yaml` had
  `REDIS_PORT={REDIS_PORT}` with no dollar sign, so `int()` received a literal brace.
- **A declared dependency was missing.** GPTCache stores in Redis, and `redis` was not in
  `requirements.txt`; the image worked only through a transitive dependency.
- **Dead configuration.** `BERT_MODEL_NAME` pointed at a component deleted in an earlier
  refactor. BERT is now a real backend again rather than a setting with nothing behind it.
- Versions are pinned, every compose variable has a default, and `/api/health` reports
  which backends are configured, so the service starts and stays useful when a key is
  absent instead of failing as a whole.


## License

MIT, see [LICENSE](LICENSE).
